# io_config.py
# I/O configuration and data access layer for filesystem, S3, and FTP sources.

import boto3
import paramiko
import pandas as pd
import json
import os
import pickle
from io import StringIO, BytesIO
from ftplib import FTP
from typing import Optional, List
from datetime import date

from oport import error_collector
from .constants import CONFIG_SOURCE_FS, CONFIG_SOURCE_FTP, CONFIG_SOURCE_S3
from .df_utils import validate_json_data, validate_dataframe_schema
from oport.fields import FIELD_DATE


class Config:
    def __init__(
        self,
        source,
        filesystem_folder,
        output_file_name,
        bucket_name,
        access_key_id,
        secret_access_key,
        s3_host,
        s3_port,
        ftp_host,
        ftp_user,
        ftp_pass,
    ):
        self.source = source
        self.output_file_name = output_file_name
        # filesystem
        self.filesystem_folder = filesystem_folder
        # s3
        self.bucket_name = bucket_name
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.s3_host = s3_host
        self.s3_port = s3_port
        # ftp
        self.ftp_host = ftp_host
        self.ftp_user = ftp_user
        self.ftp_pass = ftp_pass

    def validate(self):
        if self.source == CONFIG_SOURCE_FS:
            if self.filesystem_folder is None:
                raise ValueError("Config: filesystem folder not specified.")
        elif self.source == CONFIG_SOURCE_FTP:
            if self.ftp_host is None:
                raise ValueError("Config: ftp host not specified.")
            if self.ftp_pass is None:
                raise ValueError("Config: ftp password not specified.")
            if self.ftp_user is None:
                raise ValueError("Config: ftp user not specified.")
        elif self.source == CONFIG_SOURCE_S3:
            if self.s3_host is None:
                raise ValueError("Config: s3 host not specified.")
            if self.s3_port is None:
                raise ValueError("Config: s3 port not specified.")
            if self.access_key_id is None:
                raise ValueError("Config: access_key_id not specified.")
            if self.secret_access_key is None:
                raise ValueError("Config: secret_access_key not specified.")
        else:
            raise ValueError("Config: invalid source '{}' ".format(self.source))


def open_ftp_connection(config: Config):
    try:
        ftp = FTP(config.ftp_host)
        return ftp
    except Exception as e:
        print(
            "FTP connection error: {}, {}, {}: {}. Trying with paramiko/sftp ".format(
                config.source, config.bucket_name, config.ftp_host, e
            )
        )
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=config.ftp_host,
            username=config.ftp_user,
            password=config.ftp_pass,
            look_for_keys=False,
        )
        sftp = ssh.open_sftp()
        return sftp


def list_files(config, input_folder, prefix):
    err_collector = error_collector.get_collector()
    try:
        if config.source == "filesystem":
            files = [
                filename
                for filename in os.listdir(input_folder)
                if filename.startswith(prefix) and (not filename == prefix + ".csv")
            ]
        elif config.source == "s3":
            session = boto3.Session(
                aws_access_key_id=config.access_key_id,
                aws_secret_access_key=config.secret_access_key,
            )
            s3 = session.client(
                "s3", endpoint_url=f"http://{config.s3_host}:{config.s3_port}"
            )
            response = s3.list_objects_v2(Bucket=config.bucket_name, Prefix=prefix)
            files = [obj["Key"] for obj in response.get("Contents", [])]
            files = [f for f in files if f != prefix + ".csv"]
        elif config.source == "ftp":
            ftp = open_ftp_connection(config)
            ftp.login(user=config.ftp_user, passwd=config.ftp_pass)
            all_files = ftp.nlst()
            files = [filename for filename in all_files if filename.startswith(prefix)]
            files = [f for f in files if f != prefix + ".csv"]
            ftp.quit()
        else:
            return {"error": "Invalid source"}
        return files
    except Exception as e:
        err_collector.record_error(
            operation_name="list_files",
            error_details="No file found: {}, {}, {}, {}: {}".format(
                config.source, input_folder, config.bucket_name, config.ftp_host, e
            ),
        )
        return dict()


def read_json(config: Config, file_path: str, validate_schema: bool = False):
    try:
        err_collector = error_collector.get_collector()
        if config.source == "filesystem":
            with open(file_path, "r") as f:
                data = json.load(f)
        elif config.source == "s3":
            file_path = file_path.replace("./", "")
            session = boto3.Session(
                aws_access_key_id=config.access_key_id,
                aws_secret_access_key=config.secret_access_key,
            )
            s3 = session.client(
                "s3", endpoint_url=f"http://{config.s3_host}:{config.s3_port}"
            )
            obj = s3.get_object(Bucket=config.bucket_name, Key=file_path)
            data = json.load(obj["Body"])
        elif config.source == "ftp":
            ftp = open_ftp_connection(config)
            ftp.login(user=config.ftp_user, passwd=config.ftp_pass)
            with BytesIO() as f:
                ftp.retrbinary(f"RETR {file_path}", f.write)
                f.seek(0)
                data = json.load(f)
            ftp.quit()
        else:
            return {"error": "Invalid source"}
        if validate_schema:
            data = validate_json_data(data)
        return data
    except Exception as e:
        err_collector.record_error(
            operation_name="read_json",
            error_details="Error in reading the input stream {},{},{},{}: {}".format(
                config.source, file_path, config.bucket_name, config.ftp_host, e
            ),
        )
        return {"error": str(e)}


def read_csv(
    config: Config,
    file_path: str,
    start_date: date = None,
    end_date: date = None,
    date_field: str = FIELD_DATE,
    date_format="%Y-%m-%d",
    ignore_date=False,
    validate_schema: bool = False,
    expected_columns: List[str] = None,
):
    try:
        err_collector = error_collector.get_collector()
        date_parser = False
        if start_date is not None and end_date is not None:
            date_parser = True
            ignore_date = False

        if config.source == "filesystem":
            if not date_parser:
                if ignore_date:
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_csv(
                        file_path, parse_dates=[date_field], date_format=date_format
                    )
                    df[date_field] = df[date_field].dt.date
            else:
                chunks = pd.read_csv(
                    file_path,
                    parse_dates=[date_field],
                    date_format=date_format,
                    chunksize=5000,
                )
                df = pd.DataFrame()
                for chunk in chunks:
                    chunk[date_field] = chunk[date_field].dt.date
                    filtered_chunk = chunk[
                        (chunk[date_field] >= start_date)
                        & (chunk[date_field] <= end_date)
                    ]
                    df = pd.concat([df, filtered_chunk], ignore_index=True)

        elif config.source == "s3":
            file_path = file_path.replace("./", "")
            session = boto3.Session(
                aws_access_key_id=config.access_key_id,
                aws_secret_access_key=config.secret_access_key,
            )
            s3 = session.client(
                "s3", endpoint_url=f"http://{config.s3_host}:{config.s3_port}"
            )
            obj = s3.get_object(Bucket=config.bucket_name, Key=file_path)
            if not date_parser:
                if ignore_date:
                    df = pd.read_csv(obj["Body"])
                else:
                    df = pd.read_csv(
                        obj["Body"], parse_dates=[date_field], date_format=date_format
                    )
                    df[date_field] = df[date_field].dt.date
            else:
                chunks = pd.read_csv(
                    obj["Body"],
                    parse_dates=[date_field],
                    date_format=date_format,
                    chunksize=5000,
                )
                df = pd.DataFrame()
                for chunk in chunks:
                    chunk[date_field] = chunk[date_field].dt.date
                    filtered_chunk = chunk[
                        (chunk[date_field] >= start_date)
                        & (chunk[date_field] <= end_date)
                    ]
                    df = pd.concat([df, filtered_chunk], ignore_index=True)

        elif config.source == "ftp":
            ftp = open_ftp_connection(config)
            ftp.login(user=config.ftp_user, passwd=config.ftp_pass)
            with BytesIO() as f:
                ftp.retrbinary(f"RETR {file_path}", f.write)
                f.seek(0)
                if not date_parser:
                    if ignore_date:
                        df = pd.read_csv(f)
                    else:
                        df = pd.read_csv(
                            f, parse_dates=[date_field], date_format=date_format
                        )
                        df[date_field] = df[date_field].dt.date
                else:
                    chunks = pd.read_csv(
                        f,
                        parse_dates=[date_field],
                        date_format=date_format,
                        chunksize=5000,
                    )
                    df = pd.DataFrame()
                    for chunk in chunks:
                        chunk[date_field] = chunk[date_field].dt.date
                        filtered_chunk = chunk[
                            (chunk[date_field] >= start_date)
                            & (chunk[date_field] <= end_date)
                        ]
                        df = pd.concat([df, filtered_chunk], ignore_index=True)
            ftp.quit()
        else:
            return {"Error": "Invalid source"}

        if validate_schema and not df.empty:
            df = validate_dataframe_schema(df, expected_columns)
        return df
    except Exception as e:
        err_collector.record_error(
            operation_name="read_csv",
            error_details="Missing the input stream [{}|{}|{}|{}] ".format(
                config.source, file_path, config.bucket_name, config.ftp_host, e
            ),
        )
        return pd.DataFrame()


def read_xls(config: Config, file_path: str, validate_schema: bool = False, expected_columns: List[str] = None):
    try:
        err_collector = error_collector.get_collector()
        if config.source == "filesystem":
            df = pd.read_excel(file_path)
        elif config.source == "s3":
            file_path = file_path.replace("./", "")
            session = boto3.Session(
                aws_access_key_id=config.access_key_id,
                aws_secret_access_key=config.secret_access_key,
            )
            s3 = session.client(
                "s3", endpoint_url=f"http://{config.s3_host}:{config.s3_port}"
            )
            obj = s3.get_object(Bucket=config.bucket_name, Key=file_path)
            df = pd.read_excel(BytesIO(obj["Body"].read()))
        elif config.source == "ftp":
            ftp = open_ftp_connection(config)
            ftp.login(user=config.ftp_user, passwd=config.ftp_pass)
            with BytesIO() as f:
                ftp.retrbinary(f"RETR {file_path}", f.write)
                f.seek(0)
                df = pd.read_excel(f)
            ftp.quit()
        else:
            return {"error": "Invalid source"}

        if validate_schema and not df.empty:
            df = validate_dataframe_schema(df, expected_columns)
        return df
    except Exception as e:
        err_collector.record_error(
            operation_name="read_xls",
            error_details="Error in reading the input stream {},{},{},{}: {}".format(
                config.source, file_path, config.bucket_name, config.ftp_host, e
            ),
        )
        return pd.DataFrame()


def write_csv(config: Config, file_path: str, df: pd.DataFrame):
    try:
        err_collector = error_collector.get_collector()
        if config.source == "filesystem":
            df.to_csv(file_path, index=False)
        elif config.source == "s3":
            file_path = file_path.replace("./", "")
            session = boto3.Session(
                aws_access_key_id=config.access_key_id,
                aws_secret_access_key=config.secret_access_key,
            )
            s3 = session.client(
                "s3", endpoint_url=f"http://{config.s3_host}:{config.s3_port}"
            )
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            s3.put_object(
                Bucket=config.bucket_name, Key=file_path, Body=csv_buffer.getvalue()
            )
        elif config.source == "ftp":
            ftp = open_ftp_connection(config)
            ftp.login(user=config.ftp_user, passwd=config.ftp_pass)
            with BytesIO() as f:
                df.to_csv(f, index=False)
                f.seek(0)
                ftp.storbinary(f"STOR {file_path}", f)
            ftp.quit()
        else:
            return {"error": "Invalid source"}
        return {"status": "success"}
    except Exception as e:
        err_collector.record_error(
            operation_name="write_csv",
            error_details="Error in writing the output stream {},{},{},{}: {}".format(
                config.source, file_path, config.bucket_name, config.ftp_host, e
            ),
        )

def write_pickle(config: Config, file_path: str, obj: object):
    try:
        err_collector = error_collector.get_collector()
        if config.source == "filesystem":
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        elif config.source == "s3":
            file_path = file_path.replace("./", "")
            session = boto3.Session(
                aws_access_key_id=config.access_key_id,
                aws_secret_access_key=config.secret_access_key,
            )
            s3 = session.client(
                "s3", endpoint_url=f"http://{config.s3_host}:{config.s3_port}"
            )
            s3.put_object(
                Bucket=config.bucket_name, Key=file_path, Body=pickle.dumps(obj)
            )
        elif config.source == "ftp":
            ftp = open_ftp_connection(config)
            ftp.login(user=config.ftp_user, passwd=config.ftp_pass)
            with BytesIO(pickle.dumps(obj)) as f:
                ftp.storbinary(f"STOR {file_path}", f)
            ftp.quit()
        else:
            return {"error": "Invalid source"}
        return {"status": "success"}
    except Exception as e:
        err_collector.record_error(
            operation_name="write_pickle",
            error_details="Error in writing the pickle stream {},{},{},{}: {}".format(
                config.source, file_path, config.bucket_name, config.ftp_host, e
            ),
        )

def read_pickle(config: Config, file_path: str):
    try:
        err_collector = error_collector.get_collector()
        if config.source == "filesystem":
            with open(file_path, "rb") as f:
                return pickle.load(f)
        elif config.source == "s3":
            file_path = file_path.replace("./", "")
            session = boto3.Session(
                aws_access_key_id=config.access_key_id,
                aws_secret_access_key=config.secret_access_key,
            )
            s3 = session.client(
                "s3", endpoint_url=f"http://{config.s3_host}:{config.s3_port}"
            )
            obj_s3 = s3.get_object(Bucket=config.bucket_name, Key=file_path)
            return pickle.loads(obj_s3["Body"].read())
        elif config.source == "ftp":
            ftp = open_ftp_connection(config)
            ftp.login(user=config.ftp_user, passwd=config.ftp_pass)
            with BytesIO() as f:
                ftp.retrbinary(f"RETR {file_path}", f.write)
                f.seek(0)
                obj = pickle.load(f)
            ftp.quit()
            return obj
        else:
            raise ValueError("Invalid source")
    except Exception as e:
        err_collector.record_error(
            operation_name="read_pickle",
            error_details="Error in reading the pickle stream {},{},{},{}: {}".format(
                config.source, file_path, config.bucket_name, config.ftp_host, e
            ),
        )
        return None

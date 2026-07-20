# constants.py
# All string constants for file names, folder names, config sources and metadata paths.

# --- Default file/folder names ---
DATA_FOLDER = "."
PORTFOLIO_ASSETS_FILE = "portfolio_assets.csv"
PORTFOLIO_HOLDINGS = "holdings.csv"
PORTFOLIO_HOLDINGS_DAILY = "daily_holdings.csv"
TRANSACTION_FILE = "transactions_si.csv"
TRANSACTION_FILE_TICKERS = "transactions_si_tickers.csv"
DAILY_HOLDINGS_PREFIX = "daily_holdings_and_prices"
PREFIX_FILE_PRICE = "daily_prices_"
DAILY_PERFORMANCE = "performance.csv"

WRITE_LOCK = "write.lock"

# --- Metadata paths ---
METADATA_EXCHANCE_CODES = "./oport/metadata/exchange_to_yahoo_suffix.csv"
METADATA_CCY_TICKERS = "./oport/metadata/ccys.csv"
METADATA_ASSET_TYPES = "./oport/metadata/asset_types"
METADATA_ASSET_TYPES_HEADER_FILE = "./oport/metadata/asset_types.py"

# --- I/O source identifiers ---
CONFIG_SOURCE_FS = "filesystem"
CONFIG_SOURCE_FTP = "ftp"
CONFIG_SOURCE_S3 = "s3"

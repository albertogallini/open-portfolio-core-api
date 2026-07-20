### THIS FILE IS AUTO-GENERATED. DO NOT EDIT. ###

from openbb_core.app.static.container import Container


class Extensions(Container):
    # fmt: off
    """
Routers:
    /oport

Extensions:
    - oport@0.0.1

    - oport@0.0.1    """
    # fmt: on

    def __repr__(self) -> str:
        return self.__doc__ or ""

    @property
    def oport(self):
        # pylint: disable=import-outside-toplevel
        from . import oport

        return oport.ROUTER_oport(command_runner=self._command_runner)

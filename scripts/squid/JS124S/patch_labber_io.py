"""Monkey-patch LabberData with old functions needed by scripts in this directory.

These functions were deprecated/removed.  They're reintroduced here with minor changes
so that these old scripts can run.  They're not included in the latest LabberData
version because they're not needed by new scripts and we don't want to maintain them.
"""
from shabanipy.labber import LabberData


def patch_labberdata():
    def get_axis_dimension(self, name_or_index):
        """ Get the dimension of a sweeping channel."""
        data_attrs = self._file["Data"].attrs
        dims = {
            k: v
            for k, v in zip(data_attrs["Step index"], data_attrs["Step dimensions"])
        }
        index = self._name_or_index_to_index(name_or_index)
        if index not in dims:
            msg = (
                f"The specified axis {name_or_index} is not a stepped one. "
                f"Stepped axis are {list(dims)}."
            )
            raise ValueError(msg)
        return dims[index]

    setattr(LabberData, "get_axis_dimension", get_axis_dimension)

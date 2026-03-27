import pandera as pa
class Base_Tables(object):
    class Parcels(pa.DataFrameModel):
        parcel_id: int
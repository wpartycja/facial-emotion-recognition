from pydantic import BaseModel
# import pandas as pd

class Testing(BaseModel):
    name: str
    surname: str
# class TrackRequestBody(BaseModel):
#     id: str
#     name: str
#     popularity: float
#     duration_ms: float
#     explicit: float
#     id_artist: str
#     release_date: str
#     danceability: float
#     energy: float
#     key: float
#     loudness: float
#     speechiness: float
#     acousticness: float
#     instrumentalness: float
#     liveness: float
#     valence: float
#     tempo: float
#     time_signature: float

#     def to_df(self):
#         return pd.DataFrame({
#             "id": self.id,
#             "name": self.name,
#             "popularity": self.popularity,
#             "duration_ms": self.duration_ms,
#             "explicit": self.explicit,
#             "id_artist": self.id_artist,
#             "release_date": self.release_date,
#             "danceability": self.danceability,
#             "energy": self.energy,
#             "key": self.key,
#             "loudness": self.loudness,
#             "speechiness": self.speechiness,
#             "acousticness": self.acousticness,
#             "instrumentalness": self.instrumentalness,
#             "liveness": self.liveness,
#             "valence": self.valence,
#             "tempo": self.tempo,
#             "time_signature": self.time_signature
#         }, index=[0])


class ModeResponse(BaseModel):
    mode: str

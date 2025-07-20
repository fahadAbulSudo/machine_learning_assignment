import databases
from sqlalchemy import Column, Integer, String, Table, MetaData, create_engine
from databases import Database

DATABASE_URL = "mysql://root:Rahmani30@localhost/articledb"
engine = create_engine(DATABASE_URL)
metadata = MetaData()
Article = Table(
    "article",
     metadata,
     Column("id", Integer, primary_key = True),
     Column("title", String(100))

)

User = Table(
    "user",
     metadata,
     Column("id", Integer, primary_key = True),
     Column("username", String(100)),
     Column("password", String(200))

)

database = Database(DATABASE_URL)
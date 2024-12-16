from vanna.chromadb import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp
from vanna.ollama import Ollama
import sqlite3
import argparse
import os
import glob
import json


class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

def get_ddl(db_id) -> list:
    db_path = f'data/dev_databases/{db_id}/{db_id}.sqlite'
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]
    return tables

def delete_training_data(vn):
    ids = vn.get_training_data()['id'].values.tolist()
    for id in ids:
        vn.remove_training_data(id)


parser = argparse.ArgumentParser(prog='vanna')
parser.add_argument('database', help='database name')
parser.add_argument('-cfg', '--config', help='config for ollama', default='dev_config.json')

args = parser.parse_args()

cfg_path = glob.glob('config/{file}'.format(file=args.config))

if not cfg_path:
    raise ValueError(f"No such config file: {args.config}")
else:
    cfg_path = cfg_path.pop()

print(cfg_path)
f = open(cfg_path, 'r')
config = json.load(f)
f.close()

vn = MyVanna(config)
delete_training_data(vn)

database = args.database
vn.connect_to_sqlite(f'data/dev_databases/{database}/{database}.sqlite')
ddls = get_ddl(database)
for ddl in ddls:
    vn.train(ddl=ddl)

app = VannaFlaskApp(vn, allow_llm_to_see_data=True)
app.run()
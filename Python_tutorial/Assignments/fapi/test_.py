from fastapi.testclient import TestClient
from fastapi import FastAPI
import json
from main_2 import app


#app.include_router(users.router)

client = TestClient(app)

def test_create_user():
    res = client.post("/users/", json = {"username":"dfjhffh","password":"tyr"})
    print(res.json())
    assert res.status_code == 201

import uuid
import pytest


def unique_email():
    return f"test_{uuid.uuid4().hex[:8]}@example.com"




def test_register_valid_user(client):
    resp = client.post("/api/v1/auth/register", json={
        "email": unique_email(),
        "display_name": "Test User",
        "password": "ValidPass1!",
    })
    assert resp.status_code == 201
    assert "message" in resp.json()


def test_register_duplicate_email(client):
    email = unique_email()
    payload = {"email": email, "display_name": "User", "password": "ValidPass1!"}
    client.post("/api/v1/auth/register", json=payload)
    resp = client.post("/api/v1/auth/register", json=payload)
    assert resp.status_code == 409


def test_register_invalid_email(client):
    resp = client.post("/api/v1/auth/register", json={
        "email": "not-an-email",
        "display_name": "User",
        "password": "ValidPass1!",
    })
    assert resp.status_code == 422


def test_register_weak_password_no_uppercase(client):
    resp = client.post("/api/v1/auth/register", json={
        "email": unique_email(),
        "display_name": "User",
        "password": "weakpass1!",
    })
    assert resp.status_code == 422


def test_register_weak_password_no_digit(client):
    resp = client.post("/api/v1/auth/register", json={
        "email": unique_email(),
        "display_name": "User",
        "password": "WeakPassWord!",
    })
    assert resp.status_code == 422


def test_register_weak_password_too_short(client):
    resp = client.post("/api/v1/auth/register", json={
        "email": unique_email(),
        "display_name": "User",
        "password": "Ab1!",
    })
    assert resp.status_code == 422


def test_register_short_display_name(client):
    resp = client.post("/api/v1/auth/register", json={
        "email": unique_email(),
        "display_name": "A",
        "password": "ValidPass1!",
    })
    assert resp.status_code == 422


def test_register_missing_fields(client):
    resp = client.post("/api/v1/auth/register", json={"email": unique_email()})
    assert resp.status_code == 422




def test_login_valid_credentials(client, registered_user):
    resp = client.post("/api/v1/auth/login", json={
        "email": registered_user["email"],
        "password": registered_user["password"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_login_wrong_password(client, registered_user):
    resp = client.post("/api/v1/auth/login", json={
        "email": registered_user["email"],
        "password": "WrongPassword1!",
    })
    assert resp.status_code == 401


def test_login_nonexistent_email(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": "nobody@nowhere.com",
        "password": "SomePass1!",
    })
    assert resp.status_code == 401


def test_login_empty_body(client):
    resp = client.post("/api/v1/auth/login", json={})
    assert resp.status_code == 422




def test_get_me_valid_token(client, auth_headers, registered_user):
    resp = client.get("/api/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["email"] == registered_user["email"]
    assert data["display_name"] == registered_user["display_name"]


def test_get_me_no_token(client):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 403


def test_get_me_invalid_token(client):
    resp = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
    assert resp.status_code == 401


def test_get_me_tampered_token(client, auth_headers):
    token = auth_headers["Authorization"].split(" ")[1]
    tampered = token[:-5] + "XXXXX"
    resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tampered}"})
    assert resp.status_code == 401




def test_refresh_token_flow(client, registered_user):
    login = client.post("/api/v1/auth/login", json={
        "email": registered_user["email"],
        "password": registered_user["password"],
    })
    refresh_token = login.json()["refresh_token"]
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_used_refresh_token_rejected(client, registered_user):
    login = client.post("/api/v1/auth/login", json={
        "email": registered_user["email"],
        "password": registered_user["password"],
    })
    refresh_token = login.json()["refresh_token"]
    
    client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    
    resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 401




def test_password_hash_and_verify():
    from api.routers.auth import hash_password, verify_password
    hashed = hash_password("MyPassword1!")
    assert hashed != "MyPassword1!"
    assert verify_password("MyPassword1!", hashed) is True


def test_wrong_password_not_verified():
    from api.routers.auth import hash_password, verify_password
    hashed = hash_password("MyPassword1!")
    assert verify_password("WrongPassword1!", hashed) is False


def test_hash_is_different_each_time():
    from api.routers.auth import hash_password
    h1 = hash_password("MyPassword1!")
    h2 = hash_password("MyPassword1!")
    assert h1 != h2   
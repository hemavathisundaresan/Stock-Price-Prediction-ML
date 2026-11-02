
import streamlit as st
import json
import os
import hashlib
import secrets

USERS_FILE = "users.json"


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        600000
    ).hex()

    return f"pbkdf2_sha256${salt}${hashed}"


def verify_password(password, stored_password):
    try:
        algorithm, salt, stored_hash = stored_password.split("$")
        if algorithm != "pbkdf2_sha256":
            return False

        return secrets.compare_digest(
            hash_password(password, salt).split("$")[2],
            stored_hash
        )
    except (ValueError, AttributeError):
        return False


def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def login_page():
    set_background(
        "https://www.avanse.com/blogs/images/next-top-48.jpg"
    )
    st.title("Login Page")
    st.title("Stock Price Prediction")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    users = load_users()

    if st.button("Login"):
        stored_password = users.get(username, "")

        if verify_password(password, stored_password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.query_params["page"] = "main"
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Go to Register"):
        st.query_params["page"] = "register"
        st.rerun()


def register_page():
    set_background(
        "https://assets.forafinancial.com/blog-assets/2024/10/business-loan-docs.jpg"
    )
    st.title("Register Page")

    new_username = st.text_input("New Username")
    new_password = st.text_input(
        "New Password", type="password"
    )

    if st.button("Register"):
        if not new_username.strip() or not new_password:
            st.error("Please enter a username and password.")
            return

        users = load_users()

        if new_username in users:
            st.error("Username already exists!")
        else:
            users[new_username] = hash_password(new_password)
            save_users(users)
            st.success("Registration successful! Please login.")

    if st.button("Back to Login"):
        st.query_params["page"] = "login"
        st.rerun()


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.query_params["page"] = "login"
    st.rerun()

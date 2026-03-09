# 🧠 MindScan AI — Flask Web Application

MindScan AI is a Flask-based web application that simulates mental health and emotional analysis using AI-like logic.
It includes user authentication, chat interaction, emotion prediction, and multiple API dashboards for analysis.

---

## 🚀 Features


* 🧍‍♂️ **User Authentication** — Register, login, reset password, and manage profile.
* 💬 **AI Chat System** — Get simulated AI responses with emotional insights.
* 🖼️ **Image Prediction** — Upload images to receive emotion-based analysis.
* 📊 **Analytics Dashboards** — Includes XAI, temporal, and multimodal insights.
* ⚙️ **Privacy & Profile Management** — Update or delete your data easily.
* 📁 **SQLite Database** — Lightweight and easy to manage locally.

---


## 🧩 Folder Structure

```
project/
│
├── app.py
├── users.db
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── default-avatar.png
│
├── uploads/
│
└── user_data/
```

---

## 🛠️ Requirements

Ensure you have **Python 3.8+** installed.

Install dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
Flask==3.0.3
torch==2.3.1
torchvision==0.18.1
transformers==4.44.2
Pillow==10.3.0
langdetect==1.0.9
gunicorn==21.2.0
python-dotenv==1.0.1
```

---

## ▶️ Run the App in VS Code

1. **Open the folder in VS Code**
2. **Create a virtual environment (recommended):**

   ```bash
   
   python -m venv venv
   ```
3. **Activate the virtual environment:**

   * Windows:

     ```bash
     venv\Scripts\activate
     ```
   * macOS/Linux:

     ```bash
     source venv/bin/activate
     ```
4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Flask app:**

   ```bash
   python app.py
   ```
6. **Open the app** in your browser at:

   ```
   http://127.0.0.1:5000
   ```

---

## 🧠 Example API Endpoints

| Endpoint               | Method | Description                                   |
| ---------------------- | ------ | --------------------------------------------- |
| `/register`            | POST   | Register new user                             |
| `/login`               | POST   | Login user                                    |
| `/chat`                | POST   | Send message to AI and get emotional response |
| `/predict`             | POST   | Predict emotion from text/image               |
| `/xai-dashboard`       | GET    | Explainable AI insights                       |
| `/temporal-analysis`   | GET    | Emotion trends over time                      |
| `/multimodal-fusion`   | GET    | Combine text/image/audio insights             |
| `/performance-metrics` | GET    | Performance and accuracy data                 |
| `/privacy-settings`    | GET    | Get privacy settings                          |
| `/update-privacy`      | POST   | Update privacy preferences                    |

---

## 🧾 Example `.env` File (Optional)

If you use **python-dotenv**, create a `.env` file in the root directory:

```
FLASK_ENV=development
FLASK_APP=app.py
SECRET_KEY=super_secret_key
```

---

## 💡 Tips

* Make sure the `uploads/` and `user_data/` folders are writable.
* Run `app.py` in **Debug Mode** while developing (`debug=True` is already set).
* For production, use **Gunicorn** or **Waitress** instead of Flask’s dev server.



---

## 🧑‍💻 Author

Developed with ❤️ using **Flask**, **PyTorch**, and **Transformers**.





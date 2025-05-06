# LOreal-DSI-Multi-Model-Search-Capstone

### Environment Setup

1. **Clone the Repository**

```bash
git clone https://github.com/albertchiyh/LOreal-DSI-Multi-Model-Search-Capstone.git
cd LOreal-DSI-Multi-Model-Search-Capstone
```

---

2. **Set Up Your Environment Variables**

This project uses environment variables for sensitive credentials like API keys.
We provide a `.env-example` file as a template.

* Create your own `.env` file by copying the example:

```bash
cp .env-example .env
```

* Open `.env` and fill in your API keys or other secrets. For example:

```
OPENAI_API_KEY=your_real_key_here
```

> ⚠️ **Important:** Do **not** commit your `.env` file to Git. It's excluded via `.gitignore`.

---

3. **Install Required Packages**

We manage dependencies with `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

---

4. **Run the App**

After setting up your environment and installing dependencies, you can run the project:

```bash
streamlit run app.py
```

Or version without OpenAI API

```
streamlit run app.py
```

---



# Installation Guide - Financial Sentiment Analysis Pipeline

This guide will help you set up and run the project in VS Code, even if you have no technical experience.

## 📋 Prerequisites

### Step 1: Install Python

1. **Download Python**
   - Go to [python.org/downloads](https://www.python.org/downloads/)
   - Download Python 3.9, 3.10, or 3.11 (recommended: 3.10)
   - **Important**: During installation, check the box "Add Python to PATH"

2. **Verify Installation**
   - Open Command Prompt (Windows) or Terminal (Mac/Linux)
   - Type: `python --version`
   - You should see something like: `Python 3.10.x`

### Step 2: Install VS Code (If You Haven't Already)

1. **Download VS Code**
   - Go to [code.visualstudio.com](https://code.visualstudio.com/)
   - Download for your operating system (Windows/Mac/Linux)
   - Run the installer and follow the prompts

2. **Install Python Extension**
   - Open VS Code
   - Click the Extensions icon on the left sidebar (or press `Ctrl+Shift+X`)
   - Search for "Python"
   - Install the extension by Microsoft (it should be the first result)
   - Wait for installation to complete

---

## 🚀 Installation Steps in VS Code

### Step 3: Open Project in VS Code

1. **Download Project Files**
   - Download all project files to a folder (e.g., `C:\SentimentAnalysis` or `~/SentimentAnalysis`)
   - Make sure you have these files:
     - `main.py`
     - `config.py`
     - `data_fetching.py`
     - `sentiment_analysis.py`
     - `price_correlation.py`
     - `requirements.txt`

2. **Open Folder in VS Code**
   - Open VS Code
   - Click `File` → `Open Folder...`
   - Navigate to your project folder and click "Select Folder"
   - You should now see all your Python files in the left sidebar

### Step 4: Open Integrated Terminal

VS Code has a built-in terminal that's easier to use than Command Prompt/Terminal.

1. **Open Terminal**
   - Click `Terminal` → `New Terminal` (at the top menu)
   - Or press `` Ctrl+` `` (Control + Backtick)
   - A terminal panel will appear at the bottom of VS Code

2. **Verify Location**
   - The terminal should automatically open in your project folder
   - You should see your folder path in the terminal

### Step 5: Create Virtual Environment

A virtual environment keeps this project's packages separate from others.

1. **Create Virtual Environment**
   - In the VS Code terminal, type:
   ```bash
   python -m venv venv
   ```
   - Press Enter and wait (takes 10-30 seconds)

2. **Activate Virtual Environment**
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **On Mac/Linux:**
   ```bash
   source venv/bin/activate
   ```
   
   - Press Enter
   - You should see `(venv)` appear at the start of your terminal line
   - This means the virtual environment is active!

3. **Select Python Interpreter in VS Code**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Select Interpreter"
   - Select the one that shows `(venv)` or has `venv` in the path
   - Example: `Python 3.10.x ('venv': venv)`

### Step 6: Install Required Packages

With your virtual environment active:

```bash
pip install -r requirements.txt
```

- Press Enter
- This will take 5-10 minutes
- You'll see lots of text scrolling - this is normal!
- Wait until you see "Successfully installed..." messages
- Don't close VS Code during installation

**Troubleshooting:**
- If you see "pip: command not found", try: `python -m pip install -r requirements.txt`
- If installation fails, make sure your internet connection is stable

---

## 🔑 Step 7: Get Reddit API Credentials (Optional but Recommended)

The pipeline uses Reddit data for social sentiment. You need free API credentials.

### 7.1: Create Reddit Account
1. Go to [reddit.com](https://www.reddit.com)
2. Sign up for a free account (if you don't have one)

### 7.2: Create an App
1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Scroll down and click "create another app..."
3. Fill in:
   - **name**: `SentimentAnalysis` (or any name)
   - **app type**: Select "script"
   - **description**: (leave blank or write anything)
   - **about url**: (leave blank)
   - **redirect uri**: `http://localhost:8080`
4. Click "create app"

### 7.3: Copy Your Credentials
- You'll see your app. Under the name, there's a string of characters (like `dj2kl3jkl2j3kl`) - this is your **CLIENT_ID**
- Below "secret" is another string - this is your **CLIENT_SECRET**
- Copy both!

### 7.4: Create .env File in VS Code

1. **Create New File**
   - In VS Code, right-click in the file explorer (left sidebar)
   - Click "New File"
   - Name it exactly: `.env` (with the dot at the start)
   - Press Enter

2. **Add Your Credentials**
   - Click on the `.env` file to open it
   - Paste this and replace with your values:

```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=SentimentAnalysisPipeline/1.0
```

3. **Save the File**
   - Press `Ctrl+S` (or `Cmd+S` on Mac)

**Example .env file:**
```
REDDIT_CLIENT_ID=dj2kl3jkl2j3kl
REDDIT_CLIENT_SECRET=8fj3kl4j5lk6jlk7j
REDDIT_USER_AGENT=SentimentAnalysisPipeline/1.0
```

**Important Notes:**
- No spaces around the `=` sign
- No quotes around the values
- The file name must be exactly `.env` (some systems hide the dot)

---

## ✅ Step 8: Run the Analysis in VS Code

There are **two easy ways** to run the code in VS Code:

### Method 1: Using the Play Button (Easiest)

1. Open `main.py` in VS Code (click on it in the left sidebar)
2. Look for a **Play button (▶️)** in the top-right corner
3. Click the Play button
4. The program will run in the terminal below!

### Method 2: Using the Terminal

1. Make sure your virtual environment is active (you see `(venv)` in terminal)
2. Type in the terminal:
```bash
python main.py
```
3. Press Enter

### What to Expect:

You'll see output like this:
```
============================================================
Financial Sentiment Analysis Pipeline
Lookback Period: 2.0 years (730 days)
============================================================
Starting analysis for LLY...
Step 1/4: Fetching data for LLY...
✓ Fetched 504 price records
✓ Fetched 1250 news articles
✓ Fetched 856 social media posts
...
```

- **First time**: Takes 5-15 minutes (downloading AI models)
- **Subsequent runs**: 3-5 minutes per stock
- **Progress bars**: Show you what's happening
- Results are saved in the `results` folder (auto-created)

---

## 🎯 Step 9: View Your Results

1. **Find Results Folder**
   - Look in the left sidebar of VS Code
   - You'll see a new `results` folder
   - Click to expand it

2. **Open Result Files**
   - Click on any `.json` file to view
   - Results are in easy-to-read JSON format
   - Look for `final_classification` to see the sentiment!

3. **Example Result:**
```json
{
  "ticker": "LLY",
  "final_classification": "POSITIVE",
  "confidence_score": 0.4523,
  "sentiment_metrics": {
    "avg_sentiment": 0.2341,
    "total_return_pct": 45.67
  }
}
```

---

## 🔧 Making Changes in VS Code

### Change Stocks to Analyze

1. Open `main.py` (click it in left sidebar)
2. Find this line (around line 150):
```python
tickers = ['LLY']
```
3. Change to:
```python
tickers = ['AAPL', 'MSFT', 'GOOGL']
```
4. Save file (`Ctrl+S` or `Cmd+S`)
5. Run again with the Play button!

### Change Time Period

1. Open `config.py` (click it in left sidebar)
2. Find this line (near the top):
```python
LOOKBACK_YEARS = 2
```
3. Change to your desired period:
```python
LOOKBACK_YEARS = 5  # For 5 years
```
4. Save file (`Ctrl+S` or `Cmd+S`)
5. Run `main.py` again!

---

## 💡 VS Code Tips & Tricks

### Useful VS Code Features:

1. **Auto-save**
   - Go to `File` → `Auto Save`
   - Never worry about saving files!

2. **Split View**
   - Right-click any file → "Split Right"
   - View two files side-by-side

3. **Terminal Shortcuts**
   - `` Ctrl+` `` - Toggle terminal
   - `Ctrl+Shift+5` - Split terminal
   - Clear terminal: Type `clear` (Mac/Linux) or `cls` (Windows)

4. **Search Across Files**
   - Press `Ctrl+Shift+F`
   - Search for text across all files

5. **Zoom In/Out**
   - `Ctrl+Plus` / `Ctrl+Minus` - Adjust text size
   - Great for reading code!

### Activating Virtual Environment Each Time:

When you close VS Code and reopen:
1. Open terminal (`` Ctrl+` ``)
2. Activate venv:
   - **Windows**: `venv\Scripts\activate`
   - **Mac/Linux**: `source venv/bin/activate`
3. Look for `(venv)` in terminal
4. Now you can run `python main.py`

**Shortcut:** VS Code often activates it automatically when you open a terminal!

---

## 🐛 Troubleshooting in VS Code

### Issue: "Python not found"
**Solution:**
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose Python 3.10 from the list
4. Try running again

### Issue: "No module named 'xxx'"
**Solution:**
1. Make sure `(venv)` is visible in terminal
2. If not, activate it (see Step 5)
3. Run: `pip install -r requirements.txt` again

### Issue: "Cannot activate virtual environment"
**Windows Specific:**
1. Open PowerShell as Administrator
2. Run: `Set-ExecutionPolicy RemoteSigned`
3. Type `Y` and press Enter
4. Close and reopen VS Code
5. Try activating venv again

**Alternative for Windows:**
- Use Command Prompt instead of PowerShell
- In VS Code, click the dropdown next to `+` in terminal
- Select "Command Prompt"
- Then activate venv: `venv\Scripts\activate`

### Issue: Play button doesn't work
**Solution:**
1. Make sure Python extension is installed
2. Select Python interpreter (`Ctrl+Shift+P` → "Python: Select Interpreter")
3. Use terminal method instead: `python main.py`

### Issue: Results folder not showing
**Solution:**
1. Right-click in file explorer (left sidebar)
2. Click "Refresh"
3. Or close and reopen the folder in VS Code

### Issue: Code takes too long
**Solution:**
1. Open `config.py`
2. Change `LOOKBACK_YEARS = 0.5` (6 months - faster!)
3. Save and run again

---

## 🔄 Daily Workflow in VS Code

Once everything is set up, your daily workflow is simple:

1. **Open VS Code**
   - Open your project folder

2. **Activate Virtual Environment** (if not auto-activated)
   - Open terminal (`` Ctrl+` ``)
   - Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)

3. **Make Changes**
   - Edit `main.py` to change tickers
   - Edit `config.py` to change time period

4. **Run Analysis**
   - Click Play button ▶️ in top-right
   - Or type `python main.py` in terminal

5. **View Results**
   - Check `results` folder
   - Open `.json` files to see analysis

That's it! 🎉

---

## 🎓 Learning Resources

### VS Code Basics:
- [Official VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
- [VS Code Keyboard Shortcuts](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf)

### Python Basics:
- If you're new to Python, focus on:
  - Lists: `['AAPL', 'MSFT']`
  - Variables: `LOOKBACK_YEARS = 2`
  - Strings: `'AAPL'` or `"AAPL"`

You don't need to understand all the code - just know where to change tickers and time periods!

---

## ✅ Final Checklist

Before running your first analysis:

- [ ] Python installed and verified
- [ ] VS Code installed with Python extension
- [ ] Project folder opened in VS Code
- [ ] Virtual environment created and activated
- [ ] All packages installed (`pip install -r requirements.txt`)
- [ ] `.env` file created with Reddit credentials (optional)
- [ ] Tickers configured in `main.py`
- [ ] Time period set in `config.py`

---

## 🎉 You're All Set!

Now you can:
✅ Run analysis with one click (Play button)
✅ Edit code in VS Code's beautiful interface
✅ View results directly in VS Code
✅ Use integrated terminal for everything
✅ Split screen to compare files

**Next Steps:**
1. Read the README.md for usage examples
2. Try analyzing your first stock!
3. Experiment with different time periods
4. Analyze multiple stocks at once

**Happy Analyzing! 📊🚀**

---

## 📞 Still Need Help?

Common questions:
- **"Where's the Play button?"** - Top-right corner when viewing a `.py` file
- **"How do I know venv is active?"** - Look for `(venv)` at start of terminal line
- **"Can I use Python 3.12?"** - Stick to 3.9-3.11 for best compatibility
- **"Do I need Reddit API?"** - No, but recommended for better results

For more details on using the pipeline, see README.md!
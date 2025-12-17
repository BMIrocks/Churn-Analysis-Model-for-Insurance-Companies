# Insurance Churn Prediction Application

A full-stack application for predicting insurance customer churn using machine learning with SHAP (SHapley Additive exPlanations) analysis. The application provides real-time predictions, customer insights, and regional analytics.

## 🚀 Features

- **Customer Churn Prediction**: Predict the likelihood of customer churn with confidence scores
- **SHAP Analysis**: Understand which features contribute most to churn predictions
- **Customer Search**: Look up existing customers from the database
- **Regional Insights**: Analyze churn patterns across different geographic and demographic clusters
- **Interactive Dashboard**: Modern, responsive UI with dark/light theme support
- **Real-time Analysis**: Instant predictions and explanations

## 📋 Prerequisites

Before running this application, ensure you have the following installed:

- **Python 3.8+** (for backend API server)
- **Node.js 16+** (for frontend React application)
- **npm or yarn** (comes with Node.js)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd E-Cell-Megathon-25
```

### 2. Backend Setup (Python/Flask)

#### Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install flask flask-cors pandas numpy scikit-learn shap joblib
```

#### Verify Required Files

Make sure the following files are present in the root directory:
- `churn_model.pkl` - Trained machine learning model
- `WebApp/public/company_data.csv` - Customer database

### 3. Frontend Setup (React/TypeScript)

```bash
# Navigate to WebApp directory
cd WebApp

# Install dependencies
npm install

# Return to root directory
cd ..
```

## 🚀 Running the Application

You need to run both the backend server and frontend application simultaneously.

### Dataset Used:
https://drive.google.com/file/d/1bR3b2fygr4eGf58jNy5zF4Lkrz2gMY12/view?usp=sharing

Place this dataset inside WebApp/public/company_data.csv

### Step 1: Start the Backend API Server

Open a terminal in the project root directory:

```bash
# Activate virtual environment if not already activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Start the Flask server
python api_server.py
```

The backend server will start on **http://localhost:5000**

You should see output like:
```
✅ SHAP Analyzer initialized successfully!
📊 Loaded XXXX customers from database
🚀 Starting Flask API server on http://localhost:5000
```

### Step 2: Start the Frontend Development Server

Open a **new terminal** window and navigate to the WebApp directory:

```bash
cd WebApp

# Start the development server
npm run dev
```

The frontend will start on **http://localhost:5173** (or another port if 5173 is busy)

## 🌐 Accessing the Application

Once both servers are running:

1. Open your web browser
2. Navigate to **http://localhost:5173** (or the port shown in the terminal)
3. You should see the application home page with the Turing Finances logo

## 📖 Using the Application

### Customer Search Mode

1. Click the **LOGIN** button on the home page
2. Select **Customer Search**
3. Enter a customer ID from the database
4. View the churn prediction and SHAP analysis

### Regional Insights Mode

1. Click the **LOGIN** button on the home page
2. Select **Regional Insights**
3. Explore churn patterns across different customer segments

## 🔧 API Endpoints

The backend provides the following endpoints:

- `GET /api/health` - Health check and server status
- `GET /api/customer/<customer_id>` - Fetch customer data by ID
- `POST /api/analyze` - Analyze customer data and get predictions
- `POST /api/predict` - Alias for analyze endpoint
- `POST /api/simulate` - Simulate changes to customer parameters
- `GET /api/regional-insights` - Get regional churn analysis

## 📦 Project Structure

```
E-Cell-Megathon-25/
├── api_server.py              # Flask backend server
├── churn_model.pkl            # Trained ML model
├── model_features_37.5k_cluster_v2.pkl  # Model features
├── shap_analysis.py           # SHAP analysis utilities
├── explain.py                 # Prediction explanation
├── README.md                  # This file
└── WebApp/                    # React frontend
    ├── App.tsx                # Main application component
    ├── index.tsx              # Entry point
    ├── package.json           # Node dependencies
    ├── vite.config.ts         # Vite configuration
    ├── components/            # React components
    ├── services/              # API service layer
    ├── types.ts               # TypeScript type definitions
    └── public/
        └── company_data.csv   # Customer database
```

## 🐛 Troubleshooting

### Backend Issues

**Error: Model file not found**
- Ensure `churn_model.pkl` exists in the root directory

**Error: Company data not found**
- Verify `WebApp/public/company_data.csv` exists

**Port 5000 already in use**
- Change the port in `api_server.py` (last line): `app.run(debug=True, port=5001)`

### Frontend Issues

**Module not found errors**
- Run `npm install` in the WebApp directory

**Cannot connect to backend**
- Ensure the Flask server is running on port 5000
- Check `WebApp/services/predictionService.ts` for the correct API URL

**Port 5173 already in use**
- Vite will automatically use the next available port
- Or specify a port: `npm run dev -- --port 3000`

## 🔒 CORS Configuration

The backend is configured to accept requests from any origin. In production, update the CORS settings in `api_server.py`:

```python
CORS(app, origins=['https://your-production-domain.com'])
```

## 🏗️ Building for Production

### Frontend Build

```bash
cd WebApp
npm run build
```

The built files will be in `WebApp/dist/`

### Backend Deployment

For production deployment:
1. Use a production WSGI server (e.g., Gunicorn)
2. Set appropriate CORS origins
3. Configure environment variables
4. Use a process manager (e.g., systemd, PM2)

## 👥 Contributors

Prathmesh Sharma
Het Selarka
Aishani Sood
Vansh Goyal
Saksham Goyal



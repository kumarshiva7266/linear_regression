import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import statsmodels.api as sm
from scipy import stats
import json
from datetime import datetime

class ModernLinearRegressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Linear Regression Analysis")
        self.root.geometry("1400x900")
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10))
        self.style.configure('TLabelframe', background='#f0f0f0')
        self.style.configure('TLabelframe.Label', font=('Helvetica', 11, 'bold'))
        
        # Initialize variables
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Elastic Net': ElasticNet()
        }
        self.scaler = StandardScaler()
        self.best_model_name = None
        self.best_score = -np.inf
        self.custom_data = None
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create menu bar
        self.create_menu()
        
        # Create left panel for controls
        self.create_control_panel()
        
        # Create right panel for plots
        self.create_plot_panel()
        
        # Create bottom panel for model comparison
        self.create_comparison_panel()
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Data", command=self.import_data)
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run All Models", command=self.run_all_models)
        analysis_menu.add_command(label="Cross Validation", command=self.run_cross_validation)
        analysis_menu.add_command(label="Polynomial Features", command=self.add_polynomial_features)
        
    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Data controls
        data_frame = ttk.LabelFrame(control_frame, text="Data Settings", padding="5")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(data_frame, text="Sample Size:").grid(row=0, column=0, sticky=tk.W)
        self.sample_size = tk.StringVar(value="100")
        ttk.Entry(data_frame, textvariable=self.sample_size, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(data_frame, text="Number of Features:").grid(row=1, column=0, sticky=tk.W)
        self.feature_count = tk.StringVar(value="5")
        ttk.Entry(data_frame, textvariable=self.feature_count, width=10).grid(row=1, column=1, sticky=tk.W)
        
        # Model controls
        model_frame = ttk.LabelFrame(control_frame, text="Model Settings", padding="5")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value="Linear Regression")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=list(self.models.keys()))
        model_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Regularization controls
        reg_frame = ttk.LabelFrame(control_frame, text="Regularization", padding="5")
        reg_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(reg_frame, text="Alpha:").grid(row=0, column=0, sticky=tk.W)
        self.alpha_var = tk.StringVar(value="1.0")
        ttk.Entry(reg_frame, textvariable=self.alpha_var, width=10).grid(row=0, column=1, sticky=tk.W)
        
        # Run buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).grid(row=0, column=1, padx=5)
        
        # Results text area
        self.results_text = tk.Text(control_frame, height=10, width=40)
        self.results_text.grid(row=4, column=0, pady=5)
        
    def create_plot_panel(self):
        plot_frame = ttk.LabelFrame(self.main_frame, text="Visualizations", padding="5")
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Create figure for plots
        self.fig = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_comparison_panel(self):
        comparison_frame = ttk.LabelFrame(self.main_frame, text="Model Comparison", padding="5")
        comparison_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Create figure for model comparison
        self.comp_fig = plt.Figure(figsize=(12, 4))
        self.comp_canvas = FigureCanvasTkAgg(self.comp_fig, master=comparison_frame)
        self.comp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def import_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.custom_data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    self.custom_data = pd.read_excel(file_path)
                messagebox.showinfo("Success", "Data imported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error importing data: {str(e)}")
                
    def export_results(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                results = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model': self.model_var.get(),
                    'metrics': self.get_current_metrics(),
                    'parameters': {
                        'sample_size': self.sample_size.get(),
                        'feature_count': self.feature_count.get(),
                        'alpha': self.alpha_var.get()
                    }
                }
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=4)
                messagebox.showinfo("Success", "Results exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting results: {str(e)}")
                
    def get_current_metrics(self):
        return {
            'MAE': float(self.results_text.get("1.0", "2.0").split(":")[1].strip()),
            'MSE': float(self.results_text.get("2.0", "3.0").split(":")[1].strip()),
            'RMSE': float(self.results_text.get("3.0", "4.0").split(":")[1].strip()),
            'R2': float(self.results_text.get("4.0", "5.0").split(":")[1].strip())
        }
        
    def run_all_models(self):
        self.clear_results()
        results = {}
        
        for name, model in self.models.items():
            self.model_var.set(name)
            self.run_analysis()
            results[name] = self.get_current_metrics()
            
        self.plot_model_comparison(results)
        
    def run_cross_validation(self):
        X, y = self.generate_data()
        if X is None or y is None:
            return
            
        X_scaled = self.scaler.fit_transform(X)
        model = self.models[self.model_var.get()]
        
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        self.results_text.insert(tk.END, f"\nCross Validation Scores:\n")
        self.results_text.insert(tk.END, f"Mean CV Score: {cv_scores.mean():.4f}\n")
        self.results_text.insert(tk.END, f"CV Score Std: {cv_scores.std():.4f}\n")
        
    def add_polynomial_features(self):
        X, y = self.generate_data()
        if X is None or y is None:
            return
            
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        self.results_text.insert(tk.END, f"\nPolynomial Features:\n")
        self.results_text.insert(tk.END, f"Original features: {X.shape[1]}\n")
        self.results_text.insert(tk.END, f"Polynomial features: {X_poly.shape[1]}\n")
        
    def plot_model_comparison(self, results):
        self.comp_fig.clear()
        ax = self.comp_fig.add_subplot(111)
        
        metrics = list(results[list(results.keys())[0]].keys())
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, (model_name, model_results) in enumerate(results.items()):
            values = [model_results[metric] for metric in metrics]
            ax.bar(x + i*width, values, width, label=model_name)
            
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        self.comp_fig.tight_layout()
        self.comp_canvas.draw()
        
    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.fig.clear()
        self.comp_fig.clear()
        self.canvas.draw()
        self.comp_canvas.draw()
        
    def generate_data(self):
        try:
            if self.custom_data is not None:
                return self.custom_data.iloc[:, :-1].values, self.custom_data.iloc[:, -1].values
                
            n_samples = int(self.sample_size.get())
            n_features = int(self.feature_count.get())
            
            X = np.random.randn(n_samples, n_features)
            y = (2 * X[:, 0] + 1.5 * X[:, 1] + 0.5 * X[:, 2] + 
                 0.1 * X[:, 3] + 0.05 * X[:, 4] + np.random.randn(n_samples) * 0.5)
            return X, y
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for sample size and features")
            return None, None
    
    def run_analysis(self):
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.fig.clear()
        
        # Generate data
        X, y = self.generate_data()
        if X is None or y is None:
            return
        
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train and evaluate selected model
        model_name = self.model_var.get()
        model = self.models[model_name]
        
        # Set regularization parameter if applicable
        if hasattr(model, 'alpha'):
            model.alpha = float(self.alpha_var.get())
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        # Display results
        self.results_text.insert(tk.END, f"Results for {model_name}:\n\n")
        for metric, value in metrics.items():
            self.results_text.insert(tk.END, f"{metric}: {value:.4f}\n")
        
        # Create plots
        self.create_plots(X, y, y_test, y_pred)
        
    def create_plots(self, X, y, y_test, y_pred):
        # Create subplots
        gs = self.fig.add_gridspec(2, 2)
        
        # Residuals plot
        ax1 = self.fig.add_subplot(gs[0, 0])
        residuals = y_test - y_pred
        ax1.scatter(y_pred, residuals)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        
        # Q-Q plot
        ax2 = self.fig.add_subplot(gs[0, 1])
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        # Feature importance
        ax3 = self.fig.add_subplot(gs[1, 0])
        correlations = np.corrcoef(X.T, y)[:-1, -1]
        ax3.bar(range(len(correlations)), np.abs(correlations))
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Absolute Correlation')
        ax3.set_title('Feature Importance')
        ax3.set_xticks(range(len(correlations)))
        ax3.set_xticklabels([f'X{i+1}' for i in range(len(correlations))])
        
        # Correlation matrix
        ax4 = self.fig.add_subplot(gs[1, 1])
        df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
        df['y'] = y
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax4)
        ax4.set_title('Correlation Matrix')
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernLinearRegressionGUI(root)
    root.mainloop() 
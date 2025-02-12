import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Class for downloading the preprocessed dataset
class DatasetDownloader:
    def __init__(self, data):
        self.data = data

    def download_dataset(self):
        """Prompts the user for a filename and downloads the preprocessed dataset as a CSV file."""
        while True:
            try:
                filename = input("Enter the FILENAME you want (without .csv): ").strip()
                # Automatically append .csv if not provided
                if not filename.endswith('.csv'):
                    filename += '.csv'
                # Save the dataset as CSV
                self.data.to_csv(filename, index=False)
                print(f"Dataset saved successfully as '{filename}'.")

                # Print the additional message with line spaces
                print("\nLet's check different models and get the best model for your dataset.\n")

                break
            except Exception as e:
                print(f"An error occurred while saving the file: {e}")

class AutoML:
    def __init__(self, data_path, target_column, k_best_features=10):
        # Load and preprocess dataset
        self.data = pd.read_csv(data_path)
        self.target_column = target_column
        self.k_best_features = k_best_features
        self.X = None
        self.y = None
        self.models = {
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000)
        }
        self.best_model_name = None
        self.best_accuracy = 0

    def preprocess_data(self):
        # Convert categorical variables to numerical (if any)
        self.data = pd.get_dummies(self.data, drop_first=True)
        # Separate features and target
        self.X = self.data.drop(columns=[self.target_column]).values
        self.y = self.data[self.target_column].values

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        self.X = imputer.fit_transform(self.X)

        # Feature selection
        selector = SelectKBest(f_classif, k=self.k_best_features)
        self.X = selector.fit_transform(self.X, self.y)

        # Split dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # Download option for preprocessed data
        self.download_option()

    def download_option(self):
        """Gives the option to download the preprocessed dataset."""
        print("\nDo you want to download the preprocessed dataset? (y/n)")
        choice = input().lower().strip()
        if choice == 'y':
            downloader = DatasetDownloader(pd.DataFrame(self.X, columns=[f"Feature_{i}" for i in range(self.X.shape[1])]))
            downloader.download_dataset()

    def train_models(self):
        for name, model in self.models.items():
            # Hyperparameter tuning for Decision Tree
            if name == 'Decision Tree':
                param_grid = {
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                }
                grid_search = GridSearchCV(model, param_grid, cv=5)
                grid_search.fit(self.X_train, self.y_train)
                best_model = grid_search.best_estimator_
            else:
                # Fit the model on the training data
                model.fit(self.X_train, self.y_train)
                best_model = model
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=5)
            print(f"{name} Cross-Validation Scores: {cv_scores}")

            # Make predictions
            y_pred = best_model.predict(self.X_test)

            # Evaluate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"Accuracy ({name}): {accuracy:.4f}")

            # Output additional performance metrics
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, y_pred))

            # Check if this is the best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model_name = name

        print(f"The best model is: {self.best_model_name} with accuracy: {self.best_accuracy:.4f}")

        # Output after applying the best model on the dataset
        print("Output after applying the best model on your dataset is:")
        best_model_predictions = best_model.predict(self.X_test)
        print("Classification Report:")
        print(classification_report(self.y_test, best_model_predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, best_model_predictions))

    def run(self):
        self.preprocess_data()
        self.train_models()

# Example of how to use the AutoML class
if __name__ == "__main__":
    automl = AutoML(data_path=r'C:\Users\prana\Downloads\train.csv', target_column='Survived')
    automl.run()

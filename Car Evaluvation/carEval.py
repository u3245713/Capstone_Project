import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


class EDA:
    def __init__(self, df):
        self.df = df

    def view_data(self):
        return self.df.head()

    def attributes_values(self):
        msg = ''
        for col in self.df.columns:
            msg += f"{self.df[col].name}: {', '.join(self.df[col].unique())}."
            msg += '\n\n'
        return msg.strip()

    def missing_values(self):
        return self.df.isnull().sum()

    def pie_chart(self):
        class_counts = self.df['class values'].value_counts()
        plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
        plt.title('Value Distribution of Class Values')
        return plt

    def bar_chart(self):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()

        for i, ax in enumerate(axes):
            if i < len(self.df.columns) - 1:
                sns.countplot(x=self.df.columns[i], data=self.df, hue='class values', ax=ax)
                ax.set_title(f'{self.df.columns[i].capitalize()} Categorized by Class')
                ax.set_xlabel(f'{self.df.columns[i]}')
                ax.set_ylabel('Value Count')

        plt.tight_layout()
        return plt


class PDA:
    def __init__(self, df):
        self.df = df

    def preprocess_data(self):
        self.x = pd.get_dummies(self.df.drop(columns=['class values', 'persons', 'doors', 'lug_boot']))
        self.y = self.df['class values']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2)

    def compare_algorithms(self):
        self.num_folds = 5
        self.seed = 7
        self.scoring = 'accuracy'

        self.models = [('Decision Tree', DecisionTreeClassifier()), ('Support Vector', SVC()),
                       ('Gradient Boosting', GradientBoostingClassifier()), ('Random Forest', RandomForestClassifier())]
        self.results = []
        self.names = []
        msg = {'Model': [], 'Accuracy': [], 'Std Deviation': []}
        for self.name, self.model in self.models:
            kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            self.cv_results = cross_val_score(self.model, self.x_train, self.y_train, cv=kfold, scoring='accuracy')
            self.results.append(self.cv_results)
            self.names.append(self.name)
            msg['Model'].append(self.name)
            msg['Accuracy'].append(self.cv_results.mean())
            msg['Std Deviation'].append(self.cv_results.std())
        return pd.DataFrame(msg)

    def modelAccuracy(self):
        self.best_model = self.model
        self.best_model.fit(self.x_train, self.y_train)
        self.y_pred = self.best_model.predict(self.x_test)
        self.model_accuracy = accuracy_score(self.y_test, self.y_pred)
        msg = f"Best Model Accuracy Score on Test Set: {self.model_accuracy}"
        return msg

    def display_algorithm_comparison(self):
        self.fig = plt.figure()
        self.fig.suptitle('Algorithm Comparison')
        self.ax = self.fig.add_subplot(111)
        plt.boxplot(self.results)
        self.ax.set_xticklabels(self.names)
        return plt

    def display_classification_report(self):
        report_org = classification_report(self.y_test, self.y_pred, output_dict=True)
        report_1 = {k:v for k,v in report_org.items() if k in ('unacc', 'acc', 'good', 'vgood')}
        report_2 = {k:v for k,v in report_org.items() if k in ('macro avg', 'weighted avg')}
        accuracy = report_org['accuracy']
        df_1 = pd.DataFrame(report_1)
        df_2 = pd.DataFrame(report_2)
        msg = f'\n\n Accuracy of Testing: {accuracy}'
        return df_1, df_2, msg
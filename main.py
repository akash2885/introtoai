import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load your dataset (assuming it's in a CSV file)
# Replace 'student_spending.csv' with your actual file path
df = pd.read_csv('student_spending.csv')

# Step 1: Calculate basic financial indicators
df['total_expenses'] = df['housing'] + df['food'] + df['transportation'] + \
                       df['books_supplies'] + df['entertainment'] + \
                       df['personal_care'] + df['technology'] + \
                       df['health_wellness'] + df['miscellaneous']

df['total_income'] = df['monthly_income'] + df['financial_aid']
df['expense_to_income'] = df['total_expenses'] / df['total_income']
df['savings_rate'] = (df['total_income'] - df['total_expenses']) / df['total_income']
df['essential_expenses_ratio'] = (df['housing'] + df['food'] + df['transportation']) / df['total_income']
df['financial_aid_dependency'] = df['financial_aid'] / df['total_income']
df['discretionary_ratio'] = (df['entertainment'] + df['personal_care']) / df['total_income']

# Step 2: Generate initial vulnerability scores using a weighted formula
df['vulnerability_score'] = (
        35 * df['expense_to_income'] +
        25 * (1 - df['savings_rate']) +
        20 * df['essential_expenses_ratio'] +
        15 * df['financial_aid_dependency'] +
        5 * df['discretionary_ratio']
)

# Normalize to 0-100 scale for better interpretability
min_score = df['vulnerability_score'].min()
max_score = df['vulnerability_score'].max()
df['vulnerability_score'] = 100 * (df['vulnerability_score'] - min_score) / (max_score - min_score)

# Step 3: Prepare features (X) and target (y) for the Random Forest model
categorical_cols = ['gender', 'year_in_school', 'major', 'preferred_payment_method']
numerical_cols = ['age', 'monthly_income', 'financial_aid', 'housing', 'food',
                  'transportation', 'books_supplies', 'entertainment',
                  'personal_care', 'technology', 'health_wellness', 'miscellaneous']

# One-hot encode categorical features
X_categorical = pd.get_dummies(df[categorical_cols], drop_first=True)
X_numerical = df[numerical_cols]

# Combine all features
X = pd.concat([X_numerical, X_categorical], axis=1)
y = df['vulnerability_score']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Step 6: Make predictions on test set
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Enhanced Display Section
print("\n" + "=" * 60)
print("FINANCIAL VULNERABILITY PREDICTION MODEL")
print("=" * 60)
print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset size: {len(df)} students")
print(f"Training set: {len(X_train)} students")
print(f"Test set: {len(X_test)} students")

print("\nMODEL PERFORMANCE METRICS:")
print("-" * 30)
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Model explains {r2 * 100:.1f}% of the variance in vulnerability scores")

# Step 8: Analyze feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES:")
print("-" * 40)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['Feature']:<30} {row['Importance']:.4f}")

# Enhanced Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred, alpha=0.6, c='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Vulnerability Score')
axes[0, 0].set_ylabel('Predicted Vulnerability Score')
axes[0, 0].set_title('Actual vs Predicted Vulnerability Scores')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Feature Importance
top_features = feature_importance.head(10)
axes[0, 1].barh(top_features['Feature'], top_features['Importance'])
axes[0, 1].set_xlabel('Feature Importance')
axes[0, 1].set_title('Top 10 Feature Importance')
axes[0, 1].invert_yaxis()

# Plot 3: Residuals Distribution
residuals = y_test - y_pred
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Residuals (Actual - Predicted)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Prediction Residuals')
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)

# Plot 4: Score Distribution
axes[1, 1].hist(df['vulnerability_score'], bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Vulnerability Score')
axes[1, 1].set_ylabel('Number of Students')
axes[1, 1].set_title('Distribution of Vulnerability Scores')
axes[1, 1].axvline(30, color='green', linestyle='--', label='Low/Moderate')
axes[1, 1].axvline(60, color='red', linestyle='--', label='Moderate/High')
axes[1, 1].legend()

plt.tight_layout()
plt.show()


def generate_budget_recommendations(student_data):
    """
    Generate comprehensive budget recommendations for a student
    """
    total_income = student_data['monthly_income'] + student_data['financial_aid']

    # Calculate current expenses
    total_expenses = sum([
        student_data['housing'], student_data['food'], student_data['transportation'],
        student_data['books_supplies'], student_data['entertainment'], student_data['personal_care'],
        student_data['technology'], student_data['health_wellness'], student_data['miscellaneous']
    ])

    # Calculate recommended budget using modified 50/30/20 rule for students
    essentials_budget = total_income * 0.6  # 60% for needs
    discretionary_budget = total_income * 0.25  # 25% for wants
    savings_budget = total_income * 0.15  # 15% for savings

    # Calculate actual spending
    actual_essentials = (student_data['housing'] + student_data['food'] +
                         student_data['transportation'] + student_data['health_wellness'])
    actual_discretionary = (student_data['entertainment'] + student_data['personal_care'] +
                            student_data['technology'] + student_data['miscellaneous'])
    actual_savings = total_income - total_expenses

    return {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'recommended_essentials': essentials_budget,
        'recommended_discretionary': discretionary_budget,
        'recommended_savings': savings_budget,
        'actual_essentials': actual_essentials,
        'actual_discretionary': actual_discretionary,
        'actual_savings': actual_savings
    }


def predict_vulnerability(new_student_data, rf_model, feature_columns):
    """
    Predict financial vulnerability score for a new student with enhanced display
    """
    # Convert new student data to DataFrame
    new_df = pd.DataFrame([new_student_data])

    # Calculate financial indicators for the new student
    new_df['total_expenses'] = new_df['housing'] + new_df['food'] + new_df['transportation'] + \
                               new_df['books_supplies'] + new_df['entertainment'] + \
                               new_df['personal_care'] + new_df['technology'] + \
                               new_df['health_wellness'] + new_df['miscellaneous']

    new_df['total_income'] = new_df['monthly_income'] + new_df['financial_aid']

    # Prepare features in the same format as training data
    new_categorical = pd.get_dummies(new_df[categorical_cols], drop_first=True)
    new_numerical = new_df[numerical_cols]

    # Combine features
    new_X = pd.concat([new_numerical, new_categorical], axis=1)

    # Ensure all columns from training are present
    for col in feature_columns:
        if col not in new_X.columns:
            new_X[col] = 0

    # Keep only the columns used during training, in the same order
    new_X = new_X[feature_columns]

    # Make prediction
    vulnerability_score = rf_model.predict(new_X)[0]

    return vulnerability_score


def display_comprehensive_analysis(student_data, score, budget_info):
    """
    Display comprehensive financial analysis for a student
    """
    # Determine risk level
    if score < 30:
        risk_level = "Low Risk"
        risk_color = "green"
        risk_description = "Good financial health"
    elif score < 60:
        risk_level = "Moderate Risk"
        risk_color = "yellow"
        risk_description = "Some financial strain"
    else:
        risk_level = "High Risk"
        risk_color = "red"
        risk_description = "Significant financial vulnerability"

    print("\n" + "=" * 60)
    print("FINANCIAL VULNERABILITY ANALYSIS REPORT")
    print("=" * 60)

    # Student Information
    print(f"\nSTUDENT PROFILE:")
    print("-" * 20)
    print(f"Age: {student_data['age']} years")
    print(f"Gender: {student_data['gender']}")
    print(f"Year in School: {student_data['year_in_school']}")
    print(f"Major: {student_data['major']}")
    print(f"Payment Method: {student_data['preferred_payment_method']}")

    # Financial Summary
    print(f"\nFINANCIAL SUMMARY:")
    print("-" * 20)
    print(f"Monthly Income: ${student_data['monthly_income']:,}")
    print(f"Financial Aid: ${student_data['financial_aid']:,}")
    print(f"Total Income: ${budget_info['total_income']:,}")
    print(f"Total Expenses: ${budget_info['total_expenses']:,}")
    print(f"Net Balance: ${budget_info['actual_savings']:,.2f}")

    # Vulnerability Score
    print(f"\nVULNERABILITY ASSESSMENT:")
    print("-" * 30)
    print(f"Vulnerability Score: {score:.2f}/100")
    print(f"Risk Level: {risk_level}")
    print(f"Assessment: {risk_description}")

    # Budget Analysis
    print(f"\nBUDGET ANALYSIS:")
    print("-" * 20)
    print(f"Recommended Budget Allocation (60/25/15 rule):")
    print(f"  Essential Expenses: ${budget_info['recommended_essentials']:,.2f} (60%)")
    print(f"  Discretionary Spending: ${budget_info['recommended_discretionary']:,.2f} (25%)")
    print(f"  Savings/Emergency Fund: ${budget_info['recommended_savings']:,.2f} (15%)")

    print(f"\nActual Spending:")
    print(f"  Essential Expenses: ${budget_info['actual_essentials']:,.2f}")
    print(f"  Discretionary Spending: ${budget_info['actual_discretionary']:,.2f}")
    print(f"  Savings: ${budget_info['actual_savings']:,.2f}")

    # Detailed Expense Breakdown
    print(f"\nEXPENSE BREAKDOWN:")
    print("-" * 25)
    expenses = {
        'Housing': student_data['housing'],
        'Food': student_data['food'],
        'Transportation': student_data['transportation'],
        'Books & Supplies': student_data['books_supplies'],
        'Entertainment': student_data['entertainment'],
        'Personal Care': student_data['personal_care'],
        'Technology': student_data['technology'],
        'Health & Wellness': student_data['health_wellness'],
        'Miscellaneous': student_data['miscellaneous']
    }

    for category, amount in expenses.items():
        percentage = (amount / budget_info['total_income']) * 100
        print(f"  {category:<15} ${amount:>6,.0f} ({percentage:>4.1f}%)")

    # Recommendations
    recommendations = []

    if budget_info['actual_essentials'] > budget_info['recommended_essentials']:
        overspend = budget_info['actual_essentials'] - budget_info['recommended_essentials']
        recommendations.append(
            f"Reduce essential expenses by ${overspend:.0f} - consider roommates, meal planning, or public transportation")

    if budget_info['actual_discretionary'] > budget_info['recommended_discretionary']:
        overspend = budget_info['actual_discretionary'] - budget_info['recommended_discretionary']
        recommendations.append(
            f"Cut discretionary spending by ${overspend:.0f} - limit entertainment and subscription services")

    if budget_info['actual_savings'] < budget_info['recommended_savings']:
        shortfall = budget_info['recommended_savings'] - budget_info['actual_savings']
        recommendations.append(f"Increase savings by ${shortfall:.0f} - aim for 15% of income for emergency fund")

    if budget_info['actual_savings'] < 0:
        recommendations.append("URGENT: You're spending more than you earn! Create an emergency budget immediately")

    if score > 60:
        recommendations.append("High financial vulnerability detected - seek financial counseling services")

    print(f"\nRECOMMENDATIONS:")
    print("-" * 20)
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("Great job! Your spending is within recommended guidelines.")

    # Create visualization
    create_student_financial_visualization(student_data, budget_info, score)

    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60)


def create_student_financial_visualization(student_data, budget_info, score):
    """
    Create visualization for individual student financial analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Budget Comparison
    categories = ['Essential Expenses', 'Discretionary Spending', 'Savings']
    actual = [budget_info['actual_essentials'], budget_info['actual_discretionary'], budget_info['actual_savings']]
    recommended = [budget_info['recommended_essentials'], budget_info['recommended_discretionary'],
                   budget_info['recommended_savings']]

    x = np.arange(len(categories))
    width = 0.35

    axes[0, 0].bar(x - width / 2, actual, width, label='Actual', alpha=0.8)
    axes[0, 0].bar(x + width / 2, recommended, width, label='Recommended', alpha=0.8)
    axes[0, 0].set_ylabel('Amount ($)')
    axes[0, 0].set_title('Budget: Actual vs Recommended')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Expense Breakdown Pie Chart
    expenses = {
        'Housing': student_data['housing'],
        'Food': student_data['food'],
        'Transportation': student_data['transportation'],
        'Education': student_data['books_supplies'],
        'Entertainment': student_data['entertainment'],
        'Personal': student_data['personal_care'],
        'Technology': student_data['technology'],
        'Health': student_data['health_wellness'],
        'Other': student_data['miscellaneous']
    }

    axes[0, 1].pie(expenses.values(), labels=expenses.keys(), autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Monthly Expense Distribution')

    # Plot 3: Vulnerability Score Gauge
    vulnerability_colors = ['green', 'yellow', 'red']
    vulnerability_ranges = [0, 30, 60, 100]

    for i in range(len(vulnerability_ranges) - 1):
        axes[1, 0].barh(0, vulnerability_ranges[i + 1] - vulnerability_ranges[i],
                        left=vulnerability_ranges[i],
                        color=vulnerability_colors[i],
                        alpha=0.6,
                        height=0.8)

    axes[1, 0].scatter(score, 0, s=200, c='black', marker='|', linewidths=3)
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].set_ylim(-0.5, 0.5)
    axes[1, 0].set_xlabel('Vulnerability Score')
    axes[1, 0].set_title(f'Financial Vulnerability Score: {score:.1f}/100')
    axes[1, 0].set_yticks([])
    axes[1, 0].text(score, 0.2, f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Financial Health Summary
    metrics = {
        'Expense-to-Income': (budget_info['total_expenses'] / budget_info['total_income']) * 100,
        'Savings Rate': (budget_info['actual_savings'] / budget_info['total_income']) * 100,
        'Essential Expenses': (budget_info['actual_essentials'] / budget_info['total_income']) * 100,
        'Financial Aid Dependency': (student_data['financial_aid'] / budget_info['total_income']) * 100
    }

    axes[1, 1].bar(metrics.keys(), metrics.values())
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_title('Key Financial Metrics')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example of comprehensive analysis for a new student
new_student = {
    'age': 21,
    'gender': 'Female',
    'year_in_school': 'Junior',
    'major': 'Economics',
    'monthly_income': 1200,
    'financial_aid': 500,
    'housing': 600,
    'food': 300,
    'transportation': 100,
    'books_supplies': 200,
    'entertainment': 150,
    'personal_care': 50,
    'technology': 100,
    'health_wellness': 80,
    'miscellaneous': 50,
    'preferred_payment_method': 'Credit/Debit Card'
}

# Generate comprehensive analysis
score = predict_vulnerability(new_student, rf_model, X.columns)
budget_info = generate_budget_recommendations(new_student)
display_comprehensive_analysis(new_student, score, budget_info)
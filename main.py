import joblib
import pandas as pd

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import LocalOutlierFactor


def main():
    print('Price_Category Prediction Pipeline')

    df = pd.read_csv('data\homework.csv')
    X = df.drop('price_category', axis=1)
    y = df['price_category']#.apply(lambda x: 2.0 if x == 'high' else (1.0 if x == 'medium' else 0.0))

    def filter_data(df):
        columns_to_drop = [
            'id',
            'url',
            'region',
            'region_url',
            'price',
            'manufacturer',
            'image_url',
            'description',
            'posting_date',
            'lat',
            'long'
        ]
        return df.drop(columns_to_drop, axis=1)

    def smoothing_data(df):
        def calculate_outliers(data):
            q25 = data.quantile(0.25)
            q75 = data.quantile(0.75)
            iqr = q75 - q25
            boundaries_outliers = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
            return boundaries_outliers

        boundaries = calculate_outliers(df['year'])
        df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
        df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])
        return df

    def create_new_features(df):
        def short_model(df):
            if not pd.isna(df):
                return df.lower().split(' ')[0]
            else:
                return df
        df.loc[:, 'short_model'] = df['model'].apply(short_model)
        df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'medium'))
        return df



    numerical_selector = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_selector = make_column_selector(dtype_include=['object'])

    numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor =  ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_selector),
        ('categorical', categorical_transformer, categorical_selector)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64), max_iter=2000)
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('filter', FunctionTransformer(filter_data)),
            ('outliers', FunctionTransformer(smoothing_data)),
            ('new_features', FunctionTransformer(create_new_features)),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, '30_homework.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()



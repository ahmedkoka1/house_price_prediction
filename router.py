import pandas as pd
import joblib
from flask import Flask, render_template, request
from untils import preprocess_newinstance

# تحميل نموذج التنبؤ
model = joblib.load('xgb.pkl')

# تهيئة تطبيق Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == 'POST':
        try:
            # استقبال البيانات من النموذج
            longitude = float(request.form['longitude'])
            latitude = float(request.form['latitude'])
            median_income = float(request.form['median_income'])
            total_bedrooms = float(request.form['total_bedrooms'])
            total_rooms = float(request.form['total_rooms'])
            households = float(request.form['households'])
            population = float(request.form['population'])
            housing_median_age = float(request.form['housing_median_age'])
            ocean_proximity = request.form['ocean_proximity']

            # التأكد من عدم وجود قسمة على صفر
            if households == 0 or total_rooms == 0:
                return render_template('predict.html', prediction="Error: Invalid input values.")

            # حساب الميزات الإضافية
            rooms_per_household = total_rooms / households
            bedrooms_per_room = total_bedrooms / total_rooms
            population_per_household = population / households

            # تجهيز البيانات
            x_new = pd.DataFrame({
                'longitude': [longitude], 
                'latitude': [latitude], 
                'housing_median_age': [housing_median_age],  # ✅ تعديل الاسم ليطابق اسم العمود الأصلي
                'total_rooms': [total_rooms], 
                'total_bedrooms': [total_bedrooms],  # ✅ تعديل الاسم ليطابق العمود الناقص
                'population': [population],
                'households': [households], 
                'median_income': [median_income], 
                'ocean_proximity': [ocean_proximity], 
                'rooms_per_household': [rooms_per_household], 
                'bedrooms_per_room': [bedrooms_per_room], 
                'population_per_household': [population_per_household]
            })
            x_processing = preprocess_newinstance(x_new)

            # تنفيذ التنبؤ
            y_pred_new = model.predict(x_processing)
            prediction = '{:.4f}'.format(y_pred_new[0])

            return render_template('predict.html', prediction=prediction, longitude=longitude)

        except Exception as e:
            return render_template('predict.html', prediction=f"Error: {str(e)}")
    
    # إذا كان الطلب `GET`, قم بعرض الصفحة بدون نتيجة
    return render_template('predict.html', prediction="Enter values to predict.")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
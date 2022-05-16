from flask import Flask,render_template,request
import joblib
from helpers.SuperStore_Dummies import *
import xgboost


app=Flask(__name__)

model=joblib.load('models/model.h5')
scalar=joblib.load('models/scalar.h5')

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    all_data=request.args
    segment = str(all_data['segment'])
    region = str(all_data['region'])
    state = str(all_data['state'])
    city = str(all_data['city'])
    category = str(all_data['category'])
    sub_category = str(all_data['sub-category'])
    ship_mode = str(all_data['ship_mode'])
    discount = float(all_data['discount'])
    month = int(all_data['datetime'].split('-')[1])
    year = int(all_data['datetime'].split('-')[0])

    segment = Segment_dummies[all_data['segment']]
    region = Region_dummies[all_data['region']]
    state = State_dummies[all_data['state']]
    city = City_dummies[all_data['city']]
    category = Category_dummies[all_data['category']]
    sub_category = Sub_Category_dummies[all_data['sub-category']]
    ship_mode = Ship_Mode_dummies[all_data['ship_mode']]

    data = [discount, year, month] + segment + region + city + category + sub_category + ship_mode + state

    data_scaled = scalar.transform([data])
    pred = model.predict(data_scaled)[0]
    #return str(segment)+' '+str(all_data)
    pred = round(pred,2)
    return render_template('prediction.html', profit=pred)


if __name__=='__main__':
    app.run()
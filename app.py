import uvicorn
from fastapi import FastAPI, Body
from joblib import load

app = FastAPI(title="Milk Quality Prediction API",
              description="API for predicting milk quality",
              version="1.0")

# โหลดโมเดลและ mapping ต่างๆ
model = load('models/best_knn.pkl')  # เปลี่ยนเส้นทางไปที่ไฟล์โมเดลของคุณ
mapping = load('models/mapping.pkl')  # สมมติว่าเราใช้ mapping สำหรับ label encoder ของ 'Grade'
columns = load('models/columns.pkl')  # โหลดคอลัมน์จากไฟล์ columns.pkl

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Milk Quality Prediction API!"}

@app.post('/prediction', tags=["predictions"])
async def get_prediction(pH: float, Temperature: int, Taste: int, Odor: int, 
                         Fat: int, Turbidity: int, Colour: int):

    # รวบรวม input data ตามลำดับคอลัมน์ที่โมเดลต้องการ
    conditions = [pH, Temperature, Taste, Odor, Fat, Turbidity, Colour]
    data = []

    # แปลงข้อมูลตามที่ได้ทำ label encoding ไว้สำหรับ categorical features
    for i in columns:
        if i == 'Grade':  # ตรงนี้อาจไม่ต้องการ เพราะคุณไม่ได้มีฟีเจอร์ 'Age'
            data.append(int(conditions[columns.get_loc(i)]))
        else:
            # สำหรับฟีเจอร์ที่เป็น categorical ให้ใช้ mapping
            data.append(conditions[columns.get_loc(i)])  # ถ้าไม่ต้องการ mapping ใช้ conditions ตรงๆ

    # ทำการพยากรณ์
    prediction = model.predict([data]).tolist()
    
    # แปลงผลลัพธ์กลับมาเป็น label ที่เข้าใจได้ เช่น 'Low', 'Medium', 'High'
    prediction = list(mapping['Grade'].keys())[list(mapping['Grade'].values()).index(prediction[0])]

    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)

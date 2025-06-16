# ตรวจจับผู้ถือกล่องด้วย YOLOv11

โปรเจกต์นี้ใช้ YOLOv11 ในการตรวจจับวัตถุ โดยมุ่งเน้นที่การตรวจจับ:

- คน (person)
- กล่อง (box)

## โครงสร้างโปรเจกต์

```
BoxCarrier_Detector/
├── notebooks/
│   ├── human_box.ipynb
│   └── human_box_v2.ipynb
├── sample_data/
│   └── example_annotations.json
├── requirements.txt
├── .gitignore
└── README.md
```

## ข้อมูล (Dataset)

- จำนวนภาพที่ใช้: **58 ภาพ**
- ทำการ Annotate (ติดป้ายกำกับ) ผ่าน [Roboflow](https://roboflow.com/)
- Label ที่ใช้:
  - person (คน)
  - box (กล่อง)

> _ไฟล์ข้อมูลชุดนี้ถูกเก็บไว้ใน Google Drive และไม่ได้แนบไว้ใน repository นี้_

## สภาพแวดล้อมที่ใช้เทรนโมเดล

- ฝึกโมเดลใน **Google Colab**
- ใช้ไลบรารี YOLOv11 ผ่านแพ็กเกจ ultralytics
- ผู้ใช้งานสามารถรันได้จากไฟล์โน้ตบุ๊กในโฟลเดอร์ `notebooks/`

## วิธีใช้งาน

1. เปิดโน้ตบุ๊กผ่าน Google Colab:

   - รุ่นแรก: `notebooks/human_box.ipynb`
   - รุ่นที่สอง (ปรับปรุง): `notebooks/human_box_v2.ipynb`

2. ติดตั้งไลบรารีที่จำเป็น (ถ้ายังไม่มี):

   ```bash
   pip install -r requirements.txt
   ```

3. เชื่อมต่อกับ Google Drive เพื่อเข้าถึงข้อมูล:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. ตัวอย่างไฟล์ข้อมูลที่ส่งออกเป็น JSON อยู่ในโฟลเดอร์ `sample_data/`:

   - `sample_data/example_annotations.json`

   คุณสามารถใช้ไฟล์นี้เป็นตัวอย่างสำหรับการโหลดและตรวจสอบข้อมูล annotation ได้

## หมายเหตุ

- โปรเจกต์นี้ไม่ได้แนบไฟล์ dataset จริงไว้ใน repository เนื่องจากขนาดไฟล์และข้อจำกัดด้านลิขสิทธิ์
- กรุณาเชื่อมต่อ Google Drive ที่เก็บข้อมูล dataset ก่อนรันโน้ตบุ๊ก

---

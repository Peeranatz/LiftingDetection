# ตรวจจับผู้ถือกล่องด้วย YOLOv8

โปรเจกต์นี้ใช้ YOLOv8 ในการตรวจจับวัตถุ โดยมุ่งเน้นที่การตรวจจับ:

- คน (person)
- กล่อง (box)

##

```
BoxCarrier_Detector/
├── notebooks/
│   └── human_box.ipynb    # ไฟล์หลักสำหรับการฝึกโมเดล
├── requirements.txt
├── .gitignore
└── README.md
```

## ข้อมูล (Dataset)

- จำนวนภาพที่ใช้: **58 ภาพ**
- ทำการ Annotate (ติดป้ายกำกับ) ผ่าน [Roboflow](https://roboflow.com/)
- Label ที่ใช้:
  - `person` (คน)
  - `box` (กล่อง)

> _ไฟล์ข้อมูลชุดนี้ถูกเก็บไว้ใน Google Drive และไม่ได้แนบไว้ใน repository นี้_

## สภาพแวดล้อมที่ใช้เทรนโมเดล

- ฝึกโมเดลใน **Google Colab**
- ใช้ไลบรารี YOLOv11 ผ่านแพ็กเกจ `ultralytics`
- ผู้ใช้งานสามารถรันได้จากไฟล์ `notebooks/human_box.ipynb`

## วิธีใช้งาน

1. เปิดโน้ตบุ๊กผ่าน Google Colab:

   - `notebooks/human_box.ipynb`

2. ติดตั้งไลบรารีที่จำเป็น (ถ้ายังไม่มี):

   ```bash
   pip install -r requirements.txt
   ```

3. เชื่อมต่อกับ Google Drive เพื่อเข้าถึงข้อมูล

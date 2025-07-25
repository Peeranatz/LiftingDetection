# รายงานสัปดาห์ที่ 4: การพัฒนาโมเดลตรวจจับคนและกล่องแบบเจาะลึก

## ภาพรวมสัปดาห์ที่ 4

สัปดาห์นี้เน้นการพัฒนาโมเดลให้มีประสิทธิภาพสูงสุดผ่านการทำ Data Augmentation, การแยกเทรนโมเดลเฉพาะกลุ่ม และการ Fine-tuning แบบเป็นขั้นตอน

---

## การทดลองและผลลัพธ์ในแต่ละวัน

### **ครั้งที่1 : การเพิ่มข้อมูลด้วย Data Augmentation**

- **สมมุติฐาน:** การนำ Dataset ดีที่สุดจากสัปดาห์ที่แล้วมาทำ Data Augmentation จะช่วยเพิ่มความหลากหลายของข้อมูล ทำให้โมเดลเรียนรู้ได้ดีขึ้นและทำงานได้ในหลายสถานการณ์
- **วิธีการ:** ใช้เทคนิค Augmentation เช่น การหมุนภาพ, พลิกภาพ, ปรับแสง, เพิ่มสีสัน
- **ผลลัพธ์:** ได้ Dataset ขยายเป็น 7,500 ภาพ โมเดลแสดงความสามารถในการตรวจจับที่หลากหลายขึ้น แต่ยังต้องปรับแต่งเพิ่มเติม

### **ครั้งที่ 2 : การเพิ่มข้อมูล Negative Samples**

- **สมมุติฐาน:** การเพิ่มภาพที่ไม่ใช่คนหรือกล่อง (Negative Samples) จะช่วยลด False Positive และทำให้โมเดลแยกแยะได้ดีขึ้น
- **วิธีการ:** เพิ่มภาพพื้นหลัง, วัตถุอื่นๆ ที่ไม่เกี่ยวข้อง อย่างละ 500 ภาพ
- **ผลลัพธ์:** โมเดลลด False Positive ได้มาก ไม่ตรวจจับวัตถุที่ไม่เกี่ยวข้องเป็นคนหรือกล่อง

### **ครั้งที่ 3: การแยก Training โมเดลเฉพาะกลุ่ม**

- **สมมุติฐาน:** การแยกเทรนโมเดลสำหรับคนและกล่องแยกกัน จะทำให้แต่ละโมเดลเชี่ยวชาญในงานของตัวเอง ให้ผลลัพธ์ที่แม่นยำกว่า
- **วิธีการ:**
  - หา Dataset จาก Kaggle และ Roboflow
  - โมเดลคน: 20,000 ภาพ
  - โมเดลกล่อง: 8,000 ภาพ
- **ผลลัพธ์:** ได้โมเดลที่เชี่ยวชาญเฉพาะด้าน การตรวจจับแม่นยำขึ้นอย่างเห็นได้ชัด แต่ต้องใช้ทรัพยากรในการรันมากขึ้น

### **ครั้งที่ 4: Fine-tuning โมเดลที่เทรนในครั้งที่ 3**

- **สมมุติฐาน:** การนำโมเดลที่เทรนแยกมาทำ Fine-tuning ด้วย Dataset เดิมในสัปห์ดาก่อนหน้าจะได้โมเดลที่มีประสิทธิภาพสูงสุด
- **วิธีการ:** นำ Dataset จากสัปดาห์ก่อนมาแบ่งเป็นสองหมวด แล้วเพิ่มเข้าไปใน Fine-tuning process
- **ผลลัพธ์:** โมเดลทำงานได้ดีทั้งการจับคนแต่กล่องยังแม่นไม่เพียงพอ

### **วันจันทร์: Fine-tuning เฉพาะกล่อง**

- **สมมุติฐาน:** การ Fine-tuning เฉพาะการตรวจจับกล่องด้วย Dataset คุณภาพสูงจะช่วยแก้ปัญหาการจับกล่องที่ยังไม่สมบูรณ์
- **วิธีการ:** ใช้ Dataset จาก Kaggle ประมาณ 1,000 ภาพ ที่มีกล่องหลากหลายรูปแบบ
- **ผลลัพธ์:** การตรวจจับกล่องแม่นขึ้นโดยเฉพาะกล่องที่มีมุมมองต่างๆ แต่โมเดลขนาดใหญ่ขึ้น

### **วันอังคาร: การประเมินผลและจัดทำรายงาน**

- **กิจกรรม:** สรุปผลการทดลอง, เปรียบเทียบประสิทธิภาพ, จัดทำเอกสาร
- **ระยะเวลา:** 1 ชั่วโมง

---

## ปัญหาหลักที่เจอและวิธีแก้ไข

### 1. **การจัดการ Dataset ขนาดใหญ่**

- **ปัญหา:** Dataset 7,500 ภาพทำให้การ Training ใช้เวลานานและใช้ทรัพยากรมาก

### 2. **การสมดุลระหว่างโมเดลเฉพาะกลุ่มและโมเดลรวม**

- **ปัญหา:** โมเดลแยกให้ผลดีแต่ใช้ทรัพยากรมาก โมเดลรวมประหยัดทรัพยากรแต่อาจแม่นยำน้อยกว่า
- **วิธีแก้:** พัฒนาโมเดลแยกที่ใช้ทรัพยากรน้อยที่สุด

### 3. **การเลือก Dataset คุณภาพสูง**

- **ปัญหา:** Dataset จากแหล่งต่างๆ มีคุณภาพและมาตรฐานการ Annotation ที่แตกต่างกัน
- **วิธีแก้:** คัดเลือก Dataset อย่างพิถีพิถัน และทำการ Quality Check ก่อนนำมาใช้

---

## เทคนิคที่ใช้ในการพัฒนาโมเดล

### **Data Augmentation ขั้นสูง**

- การหมุนภาพในหลายมุม (0°, 90°, 180°, 270°)
- การปรับแต่งสี, ความสว่าง, ความคมชัด
- การเพิ่ม Noise และการเบลอภาพ
- การครอปและ Resize ในสัดส่วนต่างๆ

### **Specialized Training Strategy**

- การแยกเทรนโมเดลเฉพาะคลาส
- การ Fine-tuning แบบเป็นขั้นตอน

### **Dataset Engineering**

- การรวม Dataset จากหลายแหล่ง (Kaggle, Roboflow)
- การเพิ่ม Negative Samples เพื่อลด False Positive
- การสร้างข้อมูลสมดุลระหว่างคลาสต่างๆ

---

---

## 🔮 แผนการพัฒนาต่อไป

### **สัปดาห์ที่ 5**

- การทดสอบโมเดลในสภาพแวดล้อมจริง
- หาวิธีแก้ปัญหาตรวจจับไดไม่ต่อเนื่องและขนาดไม่เท่ากัน
  ห

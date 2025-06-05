from mongoengine import Document, StringField, DateTimeField, connect

connect("activity_db", host="localhost", port=27017)


class ObjectDetection(Document):
    person_id = StringField(required=True)  # รหัสบุคคล
    object_type = StringField(required=True)  # ประเภทวัตถุ
    start_time = DateTimeField(required=True)  # เวลาเริ่มตรวจจับ object
    end_time = DateTimeField(required=True)  # เวลาจบตรวจจับ object
    created_at = DateTimeField(required=True)  # เวลาบันทึก

    meta = {"collection": "objects"}

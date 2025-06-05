from mongoengine import Document, StringField, DateTimeField, connect

connect("activity_db", host="localhost", port=27017)


class Action(Document):
    person_id = StringField(required=True)  # รหัสบุคคล
    action = StringField(required=True)  # ชื่อ action ที่ตรวจจับได้
    start_time = DateTimeField(required=True)  # เวลาเริ่ม action
    end_time = DateTimeField(required=True)  # เวลาจบ action
    created_at = DateTimeField(required=True)  # เวลาบันทึก

    meta = {"collection": "actions"}

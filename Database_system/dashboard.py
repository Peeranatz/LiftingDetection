from flask import (
    Flask,
    render_template_string,
    request,
)  # นำเข้า Flask สำหรับสร้างเว็บ, render_template_string สำหรับแสดง HTML, request สำหรับรับค่าจากฟอร์ม

from models.action_model import Action  # นำเข้าโมเดล Action สำหรับข้อมูล action
from models.object_model import (
    ObjectDetection,
)  # นำเข้าโมเดล ObjectDetection สำหรับข้อมูล object

app = Flask(__name__)  # สร้างแอป Flask


@app.route("/", methods=["GET", "POST"])  # สร้าง route หลักของเว็บ รองรับทั้ง GET และ POST
def index():
    # รับค่าจากฟอร์ม filter (ถ้าไม่มีค่าจะเป็น string ว่าง)
    person_id = request.values.get("person_id", "")
    action = request.values.get("action", "")
    object_type = request.values.get("object_type", "")
    timestamp = request.values.get("timestamp", "")

    # สร้าง dictionary สำหรับเก็บเงื่อนไข query ของ action
    action_query = {}
    if person_id:
        action_query["person_id"] = person_id
    if action:
        action_query["action"] = action
    if timestamp:
        from datetime import datetime, timedelta

        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d")
            action_query["start_time__lte"] = dt + timedelta(days=1)
            action_query["end_time__gte"] = dt
        except Exception:
            pass

    # สร้าง dictionary สำหรับเก็บเงื่อนไข query ของ object
    object_query = {}
    if person_id:
        object_query["person_id"] = person_id
    if object_type:
        object_query["object_type"] = object_type
    if timestamp:
        from datetime import datetime, timedelta

        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d")
            object_query["start_time__lte"] = dt + timedelta(days=1)
            object_query["end_time__gte"] = dt
        except Exception:
            pass

    # ดึงข้อมูล action และ object ตามเงื่อนไข
    actions = Action.objects(**action_query).order_by("-start_time")[:20]
    objects = ObjectDetection.objects(**object_query).order_by("-start_time")[:20]

    # สร้าง match list: เหตุการณ์ที่ action กับ object ซ้อนทับกัน (person_id เดียวกันและช่วงเวลาซ้อน)
    matches = []
    for act in actions:
        for obj in objects:
            if (
                act.person_id == obj.person_id
                and act.start_time <= obj.end_time
                and act.end_time >= obj.start_time
            ):
                matches.append(
                    {
                        "person_id": act.person_id,
                        "action": act.action,
                        "object_type": obj.object_type,
                        "action_start": act.start_time,
                        "action_end": act.end_time,
                        "object_start": obj.start_time,
                        "object_end": obj.end_time,
                    }
                )

    # สร้าง HTML หน้า dashboard ด้วย Bootstrap และแสดงข้อมูลในตาราง
    return render_template_string(
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Action & Object Detection Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <style>
            body {
                background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
                min-height: 100vh;
            }
            .dashboard-container {
                max-width: 1200px;
                margin: 40px auto;
                background: #fff;
                border-radius: 18px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.08);
                padding: 32px 40px;
            }
            .table thead {
                background: #0288d1;
                color: #fff;
            }
            .table-striped>tbody>tr:nth-of-type(odd) {
                background-color: #e3f2fd;
            }
            .brand-header {
                display: flex;
                align-items: center;
                gap: 16px;
                margin-bottom: 24px;
            }
            .brand-header i {
                font-size: 2.5rem;
                color: #0288d1;
            }
            .brand-header h2 {
                margin: 0;
                font-weight: 700;
                color: #01579b;
            }
            .filter-form {
                background: #e0f7fa;
                border-radius: 12px;
                padding: 18px 24px;
                margin-bottom: 24px;
            }
            .filter-form label {
                font-weight: 500;
            }
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="brand-header">
                <i class="fa-solid fa-truck-droplet"></i>
                <h2>Action & Object Detection Dashboard</h2>
            </div>
            <!-- ฟอร์มสำหรับกรองข้อมูล (Filter Query) -->
            <form class="filter-form row g-3" method="get">
                <div class="col-md-3">
                    <label for="person_id" class="form-label">Person ID</label>
                    <input type="text" class="form-control" id="person_id" name="person_id" value="{{request.values.get('person_id','')}}">
                </div>
                <div class="col-md-3">
                    <label for="action" class="form-label">Action</label>
                    <input type="text" class="form-control" id="action" name="action" value="{{request.values.get('action','')}}">
                </div>
                <div class="col-md-3">
                    <label for="object_type" class="form-label">Object Type</label>
                    <input type="text" class="form-control" id="object_type" name="object_type" value="{{request.values.get('object_type','')}}">
                </div>
                <div class="col-md-3">
                    <label for="timestamp" class="form-label">วันที่</label>
                    <input type="date" class="form-control" id="timestamp" name="timestamp" value="{{request.values.get('timestamp','')}}">
                </div>
                <div class="col-12 d-flex align-items-end">
                    <button type="submit" class="btn btn-primary w-100"><i class="fa-solid fa-filter"></i> กรองข้อมูล</button>
                </div>
            </form>
            <div class="row">
                <div class="col-md-6">
                    <h5 class="mt-3 mb-2"><i class="fa-solid fa-person-walking"></i> Action Detection (20 รายการล่าสุด)</h5>
                    <div class="table-responsive">
                        <table class="table table-striped table-hover align-middle">
                            <thead>
                                <tr>
                                    <th>Person ID</th>
                                    <th>Action</th>
                                    <th>Start Time</th>
                                    <th>End Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for a in actions %}
                                <tr>
                                    <td>{{a.person_id}}</td>
                                    <td>{{a.action}}</td>
                                    <td>{{a.start_time.strftime('%d/%m/%Y %H:%M')}}</td>
                                    <td>{{a.end_time.strftime('%d/%m/%Y %H:%M')}}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="col-md-6">
                    <h5 class="mt-3 mb-2"><i class="fa-solid fa-box"></i> Object Detection (20 รายการล่าสุด)</h5>
                    <div class="table-responsive">
                        <table class="table table-striped table-hover align-middle">
                            <thead>
                                <tr>
                                    <th>Person ID</th>
                                    <th>Object Type</th>
                                    <th>Start Time</th>
                                    <th>End Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for o in objects %}
                                <tr>
                                    <td>{{o.person_id}}</td>
                                    <td>
                                        {% if "น้ำ" in o.object_type or "water" in o.object_type.lower() %}
                                            <i class="fa-solid fa-droplet" style="color:#039be5"></i>
                                        {% elif "ลัง" in o.object_type or "box" in o.object_type.lower() %}
                                            <i class="fa-solid fa-box" style="color:#6d4c41"></i>
                                        {% else %}
                                            <i class="fa-solid fa-cube"></i>
                                        {% endif %}
                                        {{o.object_type}}
                                    </td>
                                    <td>{{o.start_time.strftime('%d/%m/%Y %H:%M')}}</td>
                                    <td>{{o.end_time.strftime('%d/%m/%Y %H:%M')}}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <h5 class="mt-4 mb-2"><i class="fa-solid fa-link"></i> เหตุการณ์ที่ Action กับ Object ตรงกัน (Match)</h5>
            <div class="table-responsive">
                <table class="table table-bordered table-hover align-middle">
                    <thead>
                        <tr>
                            <th>Person ID</th>
                            <th>Action</th>
                            <th>Object Type</th>
                            <th>Action Start</th>
                            <th>Action End</th>
                            <th>Object Start</th>
                            <th>Object End</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for m in matches %}
                        <tr>
                            <td>{{m.person_id}}</td>
                            <td>{{m.action}}</td>
                            <td>{{m.object_type}}</td>
                            <td>{{m.action_start.strftime('%d/%m/%Y %H:%M')}}</td>
                            <td>{{m.action_end.strftime('%d/%m/%Y %H:%M')}}</td>
                            <td>{{m.object_start.strftime('%d/%m/%Y %H:%M')}}</td>
                            <td>{{m.object_end.strftime('%d/%m/%Y %H:%M')}}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """,
        actions=actions,
        objects=objects,
        matches=matches,
        request=request,
    )


if __name__ == "__main__":
    app.run(debug=True)
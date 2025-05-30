import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import torch.nn.functional as F
from twilio.rest import Client

# ✅ Twilio Configuration
ACCOUNT_SID = 'AC5fff1cf0a94dcfa130c4118f9aa98161'
AUTH_TOKEN = '65e1fa8803b39c12a4315abbd79c2b65'
TWILIO_PHONE_NUMBER = '+15073532710'
ALERT_PHONE_NUMBER = '+916369591700'

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load trained ResNet50
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 256),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, 7)
)
model.load_state_dict(torch.load("D:/Animal detection/animal_detector.pth", map_location=device), strict=False)
model.to(device)
model.eval()

# ✅ Class Labels
class_labels = {0: "Bison", 1: "Cheetah", 2: "Elephant", 3: "Men", 4: "Tiger", 5: "Wild_Boar", 6: "Women"}

# ✅ Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ SMS Control Variables
last_alert_time = datetime.min  # Stores last SMS timestamp
last_detected_animal = None  # Tracks last detected animal
ALERT_COOLDOWN = timedelta(seconds=30)  # Cooldown period (Adjust as needed)


# ✅ Send SMS Alert with Cooldown
def send_sms(animal_name, confidence):
    global last_alert_time, last_detected_animal

    current_time = datetime.now()

    # Only send SMS if cooldown has passed or a new animal is detected
    if current_time - last_alert_time > ALERT_COOLDOWN or last_detected_animal != animal_name:
        last_alert_time = current_time  # Update last alert time
        last_detected_animal = animal_name  # Update last detected animal

        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message_body = f"Alert!! {animal_name} detected with {confidence:.2f}% confidence at {current_time.strftime('%Y-%m-%d %H:%M:%S')}. Be alert!"

        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        print(f"SMS Sent: {message_body}")
    else:
        print(f"Skipping SMS: {animal_name} detected again within cooldown period.")


# ✅ Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        top_probs, top_classes = torch.topk(probabilities, 2)
        top_probs = top_probs[0].cpu().numpy() * 100
        top_classes = top_classes[0].cpu().numpy()

        class_index = top_classes[0]
        confidence = top_probs[0]

    # ✅ Send SMS if Confidence > 88% and not Men or Women
    if confidence > 88 and class_index not in [3, 6]:
        send_sms(class_labels[class_index], confidence)

    label = f"{class_labels[class_index]} ({confidence:.2f}%)"
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Animal Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
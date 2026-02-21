# 🚦 Smart Traffic Management System  
### SIH 2025 Project Submission  

---

## 📌 Project Title  
**AI-Based Smart Traffic Light Management System Using YOLOv8**

---

## 📖 Project Description  

The **Smart Traffic Management System** is an AI-powered solution designed to dynamically control traffic signals based on real-time traffic conditions. Unlike traditional fixed-timer traffic lights, this system analyzes live traffic data and adjusts signal timing intelligently.

The system uses **YOLOv8 (You Only Look Once - Version 8)** for real-time vehicle detection and traffic density estimation. Based on factors such as lane width, number of incoming vehicles, vehicle speed, and congestion level, the system decides which signal should be prioritized.

This project is developed for **Smart India Hackathon (SIH) 2025**.

---

## 🎯 Problem Statement  

Traditional traffic signals operate on fixed timers, which leads to:

- Unnecessary waiting time
- Traffic congestion
- Increased fuel consumption
- Higher pollution levels
- Poor emergency vehicle handling

Our system solves this by dynamically allocating green signal time based on real-time traffic analysis.

---

## 🚀 Key Features  

- ✅ Real-time vehicle detection using YOLOv8  
- ✅ Dynamic signal timing adjustment  
- ✅ Lane-based traffic density calculation  
- ✅ Speed-based prioritization  
- ✅ Smart decision-making algorithm  
- ✅ Scalable for smart city infrastructure  
- ✅ Emergency vehicle prioritization (Future Scope)  

---

## 🧠 System Working  

1. **Video Input Capture**
   - CCTV cameras capture live traffic feed.

2. **Vehicle Detection**
   - YOLOv8 model detects:
     - Cars
     - Bikes
     - Trucks
     - Buses

3. **Traffic Analysis**
   - Calculate:
     - Number of vehicles
     - Lane width
     - Vehicle density
     - Average vehicle speed
     - Incoming traffic rate

4. **Priority Calculation**
   - A scoring system determines which lane gets green signal priority.

5. **Dynamic Signal Allocation**
   - Green light duration is adjusted based on computed priority.

---

## 🛠️ Tech Stack  

- **YOLOv8** – Object detection model  
- **Python** – Core programming  
- **OpenCV** – Image processing  
- **NumPy / Pandas** – Data handling  
- **Flask / FastAPI** – Backend integration   

---

## 📊 Parameters Considered  

- 🚗 Number of Vehicles  
- 📏 Lane Width  
- 🚦 Incoming Traffic Flow  
- ⚡ Average Speed of Vehicles  
- 🛣️ Congestion Level  
- ⏱ Waiting Time  

---

## 🧪 Expected Outcomes  

- Reduced waiting time  
- Better traffic flow management  
- Lower fuel consumption  
- Reduced carbon emissions  
- Smart city integration  

---

## 🔮 Future Enhancements  

- Emergency vehicle detection and priority  
- Ambulance/fire truck automatic clearance  
- IoT-based signal hardware integration  
- Cloud-based traffic analytics dashboard  
- Integration with government traffic control systems  

---

## 👥 Team Members  

- Member 1 – ANMOL JENA  
- Member 2 – RAMAN BUCHHA  
- Member 3 – KUSHAAGRA SINGH  
- Member 4 – RINAV  
- Member 5 – SHAIVI PANDEY
- Member 6 – SRISHTI SINHA  

---

## 🏁 Conclusion  

This Smart Traffic Management System provides an intelligent, scalable, and efficient approach to modern traffic problems. By integrating AI-powered vehicle detection with dynamic signal control, the system aims to significantly improve urban mobility and contribute toward smart city development.

---

⭐ *Developed for Smart India Hackathon (SIH) 2025*

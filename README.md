# Vehicle detection using YOLO, SSD or Faster R-CNN

*This is a repository for my Diploma Thesis.*

## ☞ Instructions:

Before running the code adjust the following parameters:
- **VIDEO_PATH**: The path to the video file where detections will be made.<br />
- **CONFIDENCE_THRESHOLD**: The confidence threshold for detections. Values from 0.0 to 1.0.<br />
- **IOU_THRESHOLD**: The Intersection over Union (IoU) threshold used for vehicle identification between consecutive frames. Values from 0.0 to 1.0.<br />
- **NMS_THRESHOLD**: The threshold for the Non-Maximum Suppression (NMS) algorithm, which eliminates duplicate bounding boxes. Values from 0.0 to 1.0.<br />
- **BUFFER_ZONE**: The distance (in pixels) around the center of the bounding box used as a tolerance zone for vehicle counting.<br />
- **PROCESS_EVERY_NTH_FRAME**: Processes every nth frame to reduce computational load.<br />
- **SAVE_VIDEO**: Boolean that determines whether the results will be saved.<br />
- **SAVE_PATH**: The save path for the output video.<br />

## ☞ Οδηγίες:
Πριν από την εκτέλεση του κώδικα προσαρμόστε τις ακόλουθες παραμέτρους:
- **VIDEO_PATH**: Η διαδρομή προς το αρχείο βίντεο που θα γίνουν οι ανιχνεύσεις.<br />
- **CONFIDENCE_THRESHOLD**: Το όριο εμπιστοσύνης για τις ανιχνεύσεις. Δέχεται τιμές από 0.0 έως 1.0.<br />
- **IOU_THRESHOLD**: Το όριο Intersection over Union (IoU) που χρησιμοποιείται για την ταυτοποίηση ενός οχήματος μεταξύ συνεχόμενων καρέ. Δέχεται τιμές από 0.0 έως 1.0.<br />
- **NMS_THRESHOLD**: Το όριο για τον αλγόριθμο Non-Maximum Suppression (NMS), που εξαλείφει διπλότυπα bounding boxes. Δέχεται τιμές από 0.0 έως 1.0.<br />
- **BUFFER_ZONE**: Η απόσταση (σε pixels) γύρω από το κέντρο του bounding box που χρησιμοποιείται ως ζώνη ανοχής για την καταμέτρηση των οχημάτων.<br />
- **PROCESS_EVERY_NTH_FRAME**: Επιλέγεται η επεξεργασία κάθε n-οστού καρέ, ώστε να μειώνεται ο υπολογιστικός φόρτος.<br />
- **SAVE_VIDEO**: Boolean που καθορίζει αν θα αποθηκευτούν τα αποτελέσματα.<br />
- **SAVE_PATH**: Η διαδρομή αποθήκευσης για το εξαγόμενο βίντεο.<br />


# Preview

![video mp4 (1)](https://github.com/user-attachments/assets/244f220f-4864-4ea6-913e-3d11f798ea24)


Screenshot

<img width="1710" alt="1image" src="https://github.com/user-attachments/assets/c1f157f4-d480-4d87-a43d-295188a1c555" />

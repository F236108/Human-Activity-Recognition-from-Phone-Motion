I just wrapped up a deep-dive Applied Machine Learning project: **Human Activity Recognition from Raw Smartphone Motion Sensors**—writing every recurrent and attention layer **from scratch in PyTorch** (zero `nn.LSTM` or `nn.MultiheadAttention`).

Here is what we built and discovered:
- **Hand-coded architecture (46,982 parameters):** Packed 4-gate `LSTMCell`, bidirectional `BiLSTM` ($H=64$), and additive temporal `Attention` pooling over 128-step (2.56s @ 50 Hz) 9-channel IMU windows.
- **Strict person-based evaluation:** Validation split by volunteer (`[3, 15, 25, 30]`) to prevent 50%-overlap window leakage, reaching **90.74% test accuracy** across 2,947 unseen test windows (**100.00% on LAYING**, **98.94% on WALKING_UPSTAIRS**).
- **Attention interpretability:** Moving activities show sharp attention spikes (`mean max α_t` up to **0.2863**) locked onto individual step impacts, while still activities spread attention nearly flat (`0.0104–0.0166`, close to $1/128 \approx 0.0078$).
- **Zero-backend browser demo:** A single standalone `app/index.html` running the exact float32 BiLSTM + Attention weights client-side in JavaScript for live signal, probability, and attention inspection.

Check out the full walkthrough on Medium and the source code on GitHub:
- Medium Blog: <add-medium-link>
- GitHub Repository: https://github.com/F236108/Human-Activity-Recognition-from-Phone-Motion

#MachineLearning #PyTorch #DeepLearning #LSTM #Attention #TimeSeries #SignalProcessing #AppliedML

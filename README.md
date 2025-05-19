# SHAPWorkshopDevCon
## Overview
Welcome to the SHapley Additive Explanations workshop at the 2025 CIROH Dev Con conference! 

## 📂 Project Structure

### **🔹 Directory Breakdown**
📌 **`notebooks/`** → Jupyter notebooks for training models. The main workshop notebook is located in the main project directory.

📌 **`models/`** → Pretrained machine learning models and model checkpoints.

📌 **`outputs/`** → Stores generated SHAP values for the LSTM model.

📌 **`src/`** → Contains source code for the project with helper functions for data preprocessing, visualization, model building, etc.

---

## **Getting access to CIROH 2i2C JupyterHub Environment**

1. Request access to the staging environment here: ([access form](https://forms.office.com/Pages/ResponsePage.aspx?id=jnIAKtDwtECk6M5DPz-8p4IIpHdEnmhNgjOa9FjrwGtUNUoyV1UxNFIzV1AyTDhTNzdOT1Q5NVlLTC4u))'

1. Migrate to the upper right corner of the Resource page, click on "Open with..."

![alt text](https://github.com/finnmyers96/SHAPWorkshopDevCon/blob/main/images/hydroshare_resource.png?raw=true)

2. Select CUAHSI JupyterHub
![alt text](https://github.com/finnmyers96/SHAPWorkshopDevCon/blob/main/images/open_with_CUAHSIJupyterHub.png?raw=true)

3. If you are a new user, you will be prompted to request access to CUAHSI JupyterHub.
	- In the comment box, type: **"Clara, I am a SHAP Workshop participant"**. Clara from CUAHSI will be facilitating our workshop and will grant you access. If you are not participating in the workshop, this process typically 		  takes one business day.
---

##  **Setting up your Conda environment**

### **Opening the Terminal from within JupyterHub**
Once the Hydroshare resource has been opened within CUAHSI JupyterHub, open the terminal from the launcher.
![alt text](https://github.com/finnmyers96/SHAPWorkshopDevCon/blob/main/images/terminal_screenshot.png?raw=true)

### Locating the **`environment.yml` file**
Navigate to the **`config/`** directory and copy the path of the **`environment.yml`** file.
![alt text](https://github.com/finnmyers96/SHAPWorkshopDevCon/blob/main/images/yaml_file_screenshot.png?raw=true)

### Creating and activating the new Conda environment
Open the terminal and run the command **`conda env create -f <insert path to environment.yml>`** in the terminal. This will create a conda environment called **`shap-workshop`** and download the dependencies stored in the **`environment.yml`** file. Activate this environment by running **`conda activate shap-workshop`** in the terminal.

### Installing Jupyter kernel from new Conda environment 
To install a Jupyter kernel from this environment to run the shap_workshop Jupyter notebook, run the following command: **`python -m ipykernel install --user --name shap_workshop_env`**. This will create a kernel called **`shap_workshop_env`** which was installed from the **`shap-workshop`** Conda environment. 

### Selecting the kernel for the Jupyter notebook
To configure the kernel, open the shap_workshop Jupyter notebook and select the **`shap_workshop_env`** kernel that was just installed. 
![alt text](https://github.com/finnmyers96/SHAPWorkshopDevCon/blob/main/images/kernel_selection_screenshot.png?raw=true)

---

### The SHAP workshop notebook is now ready to be executed!






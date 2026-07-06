# Running a Python File with Conda (Windows) - A to Z

## 1. Open Command Prompt (CMD)

Press **Win + R**, type:

```cmd
cmd
```

Press **Enter**.

---

## 2. Check if Conda is Installed

```cmd
conda --version
```

Expected output:

```text
conda 25.x.x
```

---

## 3. If `conda` is NOT Recognized

Temporarily activate Anaconda:

```cmd
call C:\Users\saher\anaconda3\Scripts\activate.bat
```

Then verify:

```cmd
conda --version
```

---

## 4. Initialize CMD (Only Once)

If Conda still doesn't work in every CMD session:

```cmd
C:\Users\saher\anaconda3\Scripts\conda.exe init cmd.exe
```

Close CMD and open it again.

---

## 5. List Available Environments

```cmd
conda env list
```

or

```cmd
conda info --envs
```

Example:

```text
# conda environments:
#
base                  *  C:\Users\saher\anaconda3
cv                       C:\Users\saher\anaconda3\envs\cv
yolo                     C:\Users\saher\anaconda3\envs\yolo
```

---

## 6. Activate an Environment

```cmd
conda activate cv
```

Example prompt:

```text
(cv) C:\Users\saher>
```

---

## 7. Navigate to Your Project

```cmd
cd "C:\Users\saher\Desktop\Github Projects\ComputerVision\Identifying and locating flags from aerial photography"
```

> **Note:** Always use quotation marks if the path contains spaces.

---

## 8. Verify the Active Python Interpreter

```cmd
where python
```

or

```cmd
python --version
```

Example:

```text
C:\Users\saher\anaconda3\envs\cv\python.exe
```

---

## 9. Install Required Packages (Optional)

Using Conda:

```cmd
conda install numpy
```

Using Pip:

```cmd
pip install opencv-python
pip install ultralytics
pip install mavsdk
```

---

## 10. Run Your Python File

```cmd
python main.py
```

or

```cmd
python face_detection.py
```

---

## 11. Deactivate the Environment

```cmd
conda deactivate
```

---

# Common Conda Commands

## Create a New Environment

```cmd
conda create -n cv python=3.11
```

---

## Activate an Environment

```cmd
conda activate cv
```

---

## List Installed Packages

```cmd
conda list
```

---

## List All Environments

```cmd
conda env list
```

---

## Remove an Environment

```cmd
conda remove --name cv --all
```

---

## Export an Environment

```cmd
conda env export > environment.yml
```

---

## Create an Environment from a YAML File

```cmd
conda env create -f environment.yml
```

---

# Typical Workflow

```cmd
:: Open CMD

call C:\Users\saher\anaconda3\Scripts\activate.bat

conda activate cv

cd "C:\Users\saher\Desktop\Github Projects\ComputerVision\Identifying and locating flags from aerial photography"

python face_detection.py

conda deactivate
```

---

# Troubleshooting

## `conda` is Not Recognized

Check that Anaconda exists:

```cmd
dir C:\Users\saher\anaconda3
```

Temporarily activate Conda:

```cmd
call C:\Users\saher\anaconda3\Scripts\activate.bat
```

Initialize CMD permanently:

```cmd
C:\Users\saher\anaconda3\Scripts\conda.exe init cmd.exe
```

Restart Command Prompt and try again.

---

## Verify the Active Environment

```cmd
conda info
```

or

```cmd
echo %CONDA_DEFAULT_ENV%
```

Expected output:

```text
cv
```

---

## Verify the Python Executable

```cmd
where python
```

Expected output:

```text
C:\Users\saher\anaconda3\envs\cv\python.exe
```

---

# Summary

```text
Open CMD
      │
      ▼
conda activate <env_name>
      │
      ▼
cd <project_folder>
      │
      ▼
python <script_name>.py
      │
      ▼
conda deactivate
```
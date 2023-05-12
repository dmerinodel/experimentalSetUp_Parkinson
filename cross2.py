import pandas as pd
import os
import librosa
import librosa.display as disp
import numpy as np
import torch
from torch import nn
import time
import matplotlib.pyplot as plt

from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from parkinsonClassification.flow import train
import parkinsonClassification.dataug as dataug

# Antes de nada nos vamos a quedar con los nombres de los pacientes que van en
# train-test-val

# PCG
excel_path1 = 'datasets/PCG_Parkinson_ds/PCGITA_metadata.xlsx'
names1 = pd.read_excel(excel_path1)
names1 = names1[['RECORDING ORIGINAL NAME', 'SEX']]
names1['classID'] = [0 if 'C' in name else 1 for name in names1['RECORDING ORIGINAL NAME']]
# UEX
excel_path = 'datasets/UEX_Parkinson_ds/UEX_metadata.xlsx'
names = pd.read_excel(excel_path)
names = names[['Identificador', 'Genero']]
newgen = ['M' if gen == 0 else 'F' for gen in names['Genero']]  # Unificamos notación de género
names['Genero'] = newgen
names['classID'] = [0 if 'C' in name else 1 for name in names['Identificador']]
names = names.rename(columns={'Identificador': 'RECORDING ORIGINAL NAME', 'Genero': 'SEX'})

# Primero vamos a separar en hombres y mujeres
names_m = names[names['SEX'] == 'M']
names_f = names[names['SEX'] == 'F']

names1_m = names1[names1['SEX'] == 'M']
names1_f = names1[names1['SEX'] == 'F']

# Ahora vamos a extraer la muestra estratificada. Muestreamos para 90-10
# Primero nombres de entrenamiento
print('----UEX----')
train_names_m = names_m.groupby('classID', group_keys=False).apply(lambda x: x.sample(frac=1))
train_names_f = names_f.groupby('classID', group_keys=False).apply(lambda x: x.sample(frac=1))
train_names = pd.concat([train_names_m, train_names_f], ignore_index=True)

# Comprobaciones
tms = len(train_names[(train_names['SEX'] == 'M') & (train_names['classID'] == 0)])
tme = len(train_names[(train_names['SEX'] == 'M') & (train_names['classID'] == 1)])
tfs = len(train_names[(train_names['SEX'] == 'F') & (train_names['classID'] == 0)])
tfe = len(train_names[(train_names['SEX'] == 'F') & (train_names['classID'] == 1)])

print('Train: ', len(train_names))
print(f'Hombres sanos: {tms} - Hombres enfermos {tme}')
print(f'Mujeres sanas: {tfs} - Mujeres enfermas {tfe}')

print('----PCGita----')
val_names_m = names1_m.groupby('classID', group_keys=False).apply(lambda x: x.sample(frac=1))
val_names_f = names1_f.groupby('classID', group_keys=False).apply(lambda x: x.sample(frac=1))
val_names = pd.concat([val_names_m, val_names_f], ignore_index=True)

# Comprobaciones
vms = len(val_names[(val_names['SEX'] == 'M') & (val_names['classID'] == 0)])
vme = len(val_names[(val_names['SEX'] == 'M') & (val_names['classID'] == 1)])
vfs = len(val_names[(val_names['SEX'] == 'F') & (val_names['classID'] == 0)])
vfe = len(val_names[(val_names['SEX'] == 'F') & (val_names['classID'] == 1)])

print('Val: ', len(val_names))
print(f'Hombres sanos: {vms} - Hombres enfermos {vme}')
print(f'Mujeres sanas: {vfs} - Mujeres enfermas {vfe}')

# ----------------------------
# Preparando datos de entrenamiento desde los metadatos
# ----------------------------
download_path = 'datasets/UEX_Parkinson_ds/'

# Leemos archivo de metadatos
metadata_file = os.path.join(download_path, 'metadata/UEX_metadata.csv')
df = pd.read_csv(metadata_file)

# Construimos el path de los archivos añadiendo el nombre de las carpetas
df['relative_path'] = '/' + df['fold'].astype(str) + '/' + df['RECORDING_ORIGINAL_NAME'].astype(str) + '.wav'

# Nos quedamos con las columnas que importan
df = df[['relative_path', 'classID', 'sex']]

download_path1 = 'datasets/PCG_Parkinson_ds/'

# Leemos archivo de metadatos
metadata_file1 = os.path.join(download_path1, 'metadata/PCGITA_metadata.csv')
df1 = pd.read_csv(metadata_file1)

# Construimos el path de los archivos añadiendo el nombre de las carpetas
df1['relative_path'] = '/' + df1['fold'].astype(str) + '/' + df1['RECORDING_ORIGINAL_NAME'].astype(str) + '.wav'

# Nos quedamos con las columnas que importan
df1 = df1[['relative_path', 'classID', 'sex']]

# Ahora vamos a quedarnos con las grabaciones correspondientes a los nombres
# seleccionados en train, val y test.

df_train = pd.DataFrame()
for name in train_names['RECORDING ORIGINAL NAME']:
    df_train = pd.concat([df_train, df[df['relative_path'].str.contains(name)]], ignore_index=True)

df_val = pd.DataFrame()

for name in val_names['RECORDING ORIGINAL NAME']:
    df_val = pd.concat([df_val, df1[df1['relative_path'].str.contains(name)]], ignore_index=True)

# Comprobaciones.
print('Datos entrenamiento PCGita: ', len(df_train))
print('Sanos PCGita: ', len(df_train[df_train['classID'] == 0]))
print('Enfermos PCGita: ', len(df_train[df_train['classID'] == 1]))
print('-------------------------------------------')
print('Datos validación UEx: ', len(df_val))
print('Sanos PCGita: ', len(df_val[df_val['classID'] == 0]))
print('Enfermos PCGita: ', len(df_val[df_val['classID'] == 1]))

# Aumento de datos
data_path = 'datasets/UEX_Parkinson_ds/A'
dataug.make_dirs(data_path)
aug_df_train = dataug.slide_and_save(df_train, data_path)

data_val = 'datasets/PCG_Parkinson_ds/A'
dataug.make_dirs(data_val)
aug_df_val = dataug.slide_and_save(df_val, data_val)

# Recuperamos los nombres de cada audio, necesarios para generar los espectrogramas
names_aug_train = aug_df_train['relative_path'].str.split('/', expand=True)[2]
names_aug_val = aug_df_val['relative_path'].str.split('/', expand=True)[2]

# ------------------------------------------------------------
# Creamos las carpetas donde vamos a organizar las imágenes
# ------------------------------------------------------------
ruta_imgs = 'datasets/CROSS2/organized'

os.makedirs(os.path.join(ruta_imgs, 'train', 'control'))
os.makedirs(os.path.join(ruta_imgs, 'train', 'patologicas'))
os.makedirs(os.path.join(ruta_imgs, 'valid', 'control'))
os.makedirs(os.path.join(ruta_imgs, 'valid', 'patologicas'))

ruta_img_train_c = os.path.join(ruta_imgs, 'train', 'control')
ruta_img_train_p = os.path.join(ruta_imgs, 'train', 'patologicas')
ruta_img_val_c = os.path.join(ruta_imgs, 'valid', 'control')
ruta_img_val_p = os.path.join(ruta_imgs, 'valid', 'patologicas')

# -------------------------------------------------------------------
# Generamos los espectrogramas de entrenamiento
# -------------------------------------------------------------------
new_data_path = os.path.join(data_path, 'aug')
inicio = time.time()
for name in names_aug_train:
    # Accedemos al path relativo y lo guardamos como string
    rel_path = aug_df_train[aug_df_train['relative_path'].str.contains(name)]['relative_path'].astype(str).values[0][1:]
    ruta_aud = os.path.join(new_data_path, rel_path)

    name_img = name.split('.')[0] + '.png'

    if 'control' in ruta_aud:
        ruta_img = os.path.join(ruta_img_train_c, name_img)
    else:
        ruta_img = os.path.join(ruta_img_train_p, name_img)

    # Leemos y generamos espectrograma
    samples, sample_rate = librosa.load(ruta_aud, sr=44100)
    sgram = librosa.stft(samples)
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate, n_fft=1024, n_mels=256)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

    # Pintamos para guardar
    fig, ax = plt.subplots()
    img = disp.specshow(mel_sgram, sr=sample_rate, ax=ax)
    ax.set_axis_off()
    fig.savefig(ruta_img, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
final = time.time()
print(f'Tiempo para generar los espectrogramas de entrenamiento: {(final - inicio) / 60}')

# -------------------------------------------------------------------
# Generamos los espectrogramas de validación
# -------------------------------------------------------------------
new_data_val = os.path.join(data_val, 'aug')
inicio = time.time()
for name in names_aug_val:
    # Accedemos al path relativo y lo guardamos como string
    rel_path = aug_df_val[aug_df_val['relative_path'].str.contains(name)]['relative_path'].astype(str).values[0][1:]
    if 'w8' in rel_path:
        ruta_aud = os.path.join(new_data_val, rel_path)
        name_img = name.split('.')[0] + '.png'

        if 'control' in ruta_aud:
            ruta_img = os.path.join(ruta_img_val_c, name_img)
        else:
            ruta_img = os.path.join(ruta_img_val_p, name_img)

        # Leemos y generamos espectrograma
        samples, sample_rate = librosa.load(ruta_aud, sr=44100)
        sgram = librosa.stft(samples)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate, n_fft=1024, n_mels=256)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

        # Pintamos para guardar
        fig, ax = plt.subplots()
        img = disp.specshow(mel_sgram, sr=sample_rate, ax=ax)
        ax.set_axis_off()
        fig.savefig(ruta_img, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
final = time.time()
print(f'Tiempo para generar los espectrogramas de validación {(final-inicio)/60}')
# ---------------------------------------------------------------------------
# Para que las imágenes puedan entrar al modelo ResNet hay que transformarlas
# ---------------------------------------------------------------------------
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# -------------------------------------------------------------------
# Ahora creamos los datasets leyendo las imágenes y transformándolas
# -------------------------------------------------------------------
new_root = ruta_imgs

# Carga
train_data = datasets.ImageFolder(new_root + '/train', transform)
valid_data = datasets.ImageFolder(new_root + '/valid', transform)

batch_size = 64

# Convertir para el modelo
dataloaders = {'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
               'valid': DataLoader(valid_data, batch_size=batch_size, shuffle=True)}

# ---------------------------------------------------------------
# Importamos el modelo con los pesos preentrenados en ImageNet
# ---------------------------------------------------------------
#myModel = resnet50(weights=ResNet50_Weights.DEFAULT)
myModel = resnet50()

# Nuevo MLP del final
mlp = [nn.Linear(in_features=2048, out_features=1024, bias=True), nn.Dropout(0.5), nn.PReLU(),
       nn.Linear(in_features=1024, out_features=512, bias=True), nn.Dropout(0.5), nn.PReLU(),
       nn.Linear(in_features=512, out_features=2, bias=True), nn.Dropout(0.5), nn.PReLU()]
mlp = nn.Sequential(*mlp)

# La modificamos
myModel.fc = mlp

# CARGAMOS PESOS PRETRAIN
weights = torch.load('models/pretrain.pt')
myModel.load_state_dict(weights)

# Definimos la función de salida y el algoritmo de aprendizaje
criterion = nn.CrossEntropyLoss()

# Diferentes learning rates para cada parte.
encoder = []
decoder = []
for name, param in myModel.named_parameters():
    if 'fc' in name:
        decoder.append(param)
    else:
        encoder.append(param)
# Pasamos por separados los learning rates.
L_RATE1 = 5e-4
L_RATE2 = 8e-3
DEFOULT_LR = 1e-2
optimizer = torch.optim.SGD([{'params': encoder}, {'params': decoder}], lr=DEFOULT_LR, momentum=0.95)
optimizer.param_groups[0]['lr'] = L_RATE1
optimizer.param_groups[1]['lr'] = L_RATE2

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# -------------------------
# Entrenamos
# -------------------------
use_cuda = torch.cuda.is_available()
n_epochs = 100
print('GPU:', use_cuda)
inicio = time.time()
loss_dict, model = train(n_epochs=n_epochs,
                         dataloaders=dataloaders,
                         model=myModel,
                         criterion=criterion,
                         use_cuda=use_cuda,
                         optimizer=optimizer)
final = time.time()
train_time = (final - inicio) / 60
print(f'Ha tardado {train_time} minutos.')

fig = plt.figure(figsize=(16, 9))
plt.plot(loss_dict["valid"])
plt.plot(loss_dict["train"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["valid loss", "training loss"])
plt.savefig("figures/cross2_loss.png")
plt.close()

fig = plt.figure(figsize=(16, 9))
plt.plot(loss_dict["valid_acc"])
plt.plot(loss_dict["train_acc"])
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend(["valid accuracy", "train accuracy"])
plt.savefig("figures/cross2_acc.png")
plt.close()

res = open('results/cross2.txt', 'a')
res.write(f'Tiempo: {train_time} minutos\n')
res.write('train_loss \t valid_loss \t train_acc \t valid_acc\n')
for dat1, dat2, dat3, dat4 in zip(loss_dict['train'], loss_dict['valid'],
                                  loss_dict['train_acc'], loss_dict['valid_acc']):
    res.write(f'{dat1} \t {dat2} \t {dat3} \t {dat4}\n')
res.close()

import subprocess
import os

folder = '/home/s233039/Desktop/KU/data/69/microelectrode_experiments/'
#81_MicroElect_03102024_astrocyte_jrgeco1a_ttx_nifedipine
for file in os.listdir(folder):
    if file.endswith(".oir"):
        print(file)
        command = F'/home/s233039/Desktop/KU/bftools/bfconvert "{os.path.join(folder, file)}" "{os.path.join(folder, file[:-3]+"tiff")}"'
        print(command)
        subprocess.run(command, shell = True)
   
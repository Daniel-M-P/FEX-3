import matplotlib.pyplot as plt
import numpy as np

# Datos
I = np.array([0.13, 0.29, 0.45, 0.61, 0.76])       
VDC = np.array([0.108, 0.108, 0.108, 0.108, 0.108])   
VAC = np.array([0.49, 1.09, 1.64, 2.25, 2.79])        
# Ajuste lineal VDC
m_dc, b_dc = np.polyfit(I, VDC, 1)   
VDC_fit = m_dc * I + b_dc

# Ajuste lineal VAC
m_ac, b_ac = np.polyfit(I, VAC, 1)
VAC_fit = m_ac * I + b_ac

# Errores estimados
I_err = np.array([0.01]*5)
VDC_err = np.array([0.002]*5)
VAC_err = np.array([0.02]*5)

# VDC vs Corriente 
plt.figure()
plt.errorbar(I, VDC, xerr=I_err, yerr=VDC_err, fmt='o', color='blue', label='Datos VDC', capsize=4)
plt.plot(I, VDC_fit, '-', color='red', label=f'Ajuste lineal: {m_dc:.3e}·I + {b_dc:.3e}')
plt.xlabel("Corriente I (A)")
plt.ylabel("VDC (V)")
plt.ylim(0, 0.200)
plt.grid(True)
plt.legend(fontsize='small')
plt.tight_layout()
plt.show()

# VAC vs Corriente
plt.figure()
plt.errorbar(I, VAC, xerr=I_err, yerr=VAC_err, fmt='o', color='orange', label='Datos', capsize=4)
plt.plot(I, VAC_fit, '-', color='red', label=f'Ajuste lineal: {m_ac:.3e}·I + {b_ac:.3e}')
plt.xlabel("Corriente I (A)")
plt.ylabel("VAC (V)")
plt.grid(True)
plt.legend(fontsize='small')
plt.tight_layout()
plt.show()

# Pendientes
print("Pendientes calculadas:")
print(f" - VDC vs I: m = {m_dc:.3e} V/A")
print(f" - VAC vs I: m = {m_ac:.3e} V/A")

# Calcular R^2 para VAC vs I
VAC_fit = m_ac * I + b_ac
ss_res = np.sum((VAC - VAC_fit)**2)                
ss_tot = np.sum((VAC - np.mean(VAC))**2)           
R2_vac = 1 - (ss_res / ss_tot)

print(f"Coeficiente de determinación R² = {R2_vac:.4f}")

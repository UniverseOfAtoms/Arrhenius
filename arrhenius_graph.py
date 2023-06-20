import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

rate_window = [2.5, 5, 10, 20, 50, 100, 200, 500, 1000]

# Defect EH1
temp_EH1 = [188.5, 194.4, 198.24, 204.31, 209.3, 217.54, 224.81, 230.74, 239.23]
_kBT_EH1 = [round((1/(1.38e-23*item))*1.6e-19, 2) for item in temp_EH1]
ln_e_T2_EH1 = [round(np.log(rate_window[i]/temp_EH1[i]**2), 2) for i in range(len(temp_EH1))]

# Defect Z12
temp_Z12 = [289.51, 296.53, 304.91, 312.09, 321.43, 333.2, 341.1, 350.97, 359.92]
_kBT_Z12 = [round((1/(1.38e-23*item))*1.6e-19, 2) for item in temp_Z12]
ln_e_T2_Z12 = [round(np.log(rate_window[i]/temp_Z12[i]**2), 2) for i in range(len(temp_Z12))]

# Defect EH3
_kBT_EH3 = [35.77022, 35.57377, 34.07955, 32.94931, 32.45494, 31.97519, 29.63147, 30.05003, 28.14947]
ln_e_T2_EH3 = [-10.646, -9.96387, -9.35654, -8.73085, -7.84479, -7.18143, -6.64053, -5.69619, -5.13371]

# M-centers
temp_M1 = [188.28, 190.57, 197.42, 202, 208.85, 215.71, 222.57, 229.42, 236.28]
temp_M2 = [288.85, 298, 303.71, 315.14, 320.85, 325.42]
temp_M3 = [322, 326.57, 330, 336.85, 351.71]
kBT_M1 = [round((1/(1.38e-23*item))*1.6e-19, 2) for item in temp_M1]
kBT_M2 = [round((1/(1.38e-23*item))*1.6e-19, 2) for item in temp_M2]
kBT_M3 = [round((1/(1.38e-23*item))*1.6e-19, 2) for item in temp_M3]
ln_e_T2_M1 = [round(np.log(rate_window[i]/temp_M1[i]**2), 2) for i in range(len(temp_M1))]
ln_e_T2_M2 = [round(np.log(rate_window[i]/temp_M2[i]**2), 2) for i in range(len(temp_M2))]
ln_e_T2_M3 = [round(np.log(rate_window[i]/temp_M3[i]**2), 2) for i in range(len(temp_M3))]

kbT = [_kBT_EH1, _kBT_Z12, _kBT_EH3]
ln_e_T2 = [ln_e_T2_EH1, ln_e_T2_Z12, ln_e_T2_EH3]

kbT_M = [kBT_M1, kBT_M2, kBT_M3]
ln_e_T2_M = [ln_e_T2_M1, ln_e_T2_M2, ln_e_T2_M3]


model = LinearRegression()
model.fit(np.array(_kBT_EH1).reshape(-1, 1), ln_e_T2_EH1)
slope_EH1 = model.coef_
intercept_EH1 = model.intercept_
model.fit(np.array(_kBT_Z12).reshape(-1, 1), ln_e_T2_Z12)
slope_Z12 = model.coef_
intercept_Z12 = model.intercept_
model.fit(np.array(_kBT_EH3).reshape(-1, 1), ln_e_T2_EH3)
slope_EH3 = model.coef_
intercept_EH3 = model.intercept_

model.fit(np.array(kBT_M1).reshape(-1, 1), ln_e_T2_M1)
slope_M1 = model.coef_
intercept_M1 = model.intercept_
model.fit(np.array(kBT_M2).reshape(-1, 1), ln_e_T2_M2)
slope_M2 = model.coef_
intercept_M2 = model.intercept_
model.fit(np.array(kBT_M3).reshape(-1, 1), ln_e_T2_M3)
slope_M3 = model.coef_
intercept_M3 = model.intercept_


intercepts = [intercept_EH1, intercept_Z12, intercept_EH3]
slopes = [slope_EH1, slope_Z12, slope_EH3]
defects = ["EH1", "Z12", "EH3"]

intercepts_M = [intercept_M1, intercept_M2, intercept_M3]
slopes_M = [slope_M1, slope_M2, slope_M3]
m_centers = ["M1", "M2", "M3"]

fig, ax = plt.subplots()
for i in range(len(ln_e_T2)):
    ax.scatter(kbT[i], ln_e_T2[i], label=f"{defects[i]}")
    plt.plot(kbT[i], intercepts[i]+slopes[i]*kbT[i])
ax.tick_params(direction='in', right=True, left=True, bottom=True, top=True)
ax.set_xlabel("1/kBT (eV-1)")
ax.set_ylabel("ln(e/T2 * 1s-1K-2)")
ax.set_xlim(28, 64)
ax.legend()


fig1, ax1 = plt.subplots()
for i in range(len(ln_e_T2_M)):
    ax1.scatter(kbT_M[i], ln_e_T2_M[i], label=f"{m_centers[i]}")
    plt.plot(kbT_M[i], intercepts_M[i]+slopes_M[i]*kbT_M[i])
ax1.tick_params(direction='in', right=True, left=True, bottom=True, top=True)
ax1.set_xlabel("1/kBT (eV-1)")
ax1.set_ylabel("ln(e/T2 * 1s-1K-2)")
ax1.set_xlim(28, 64)
ax1.legend()
plt.show()

# Activation energy for defects
Ea = [-slope[0] for slope in slopes]
Ea_m = [-slope[0] for slope in slopes_M]
for i in range(len(Ea)):
    print(f"Activation energy for defect {defects[i]}: {Ea[i]}")
    print(f"Activation energy for M-center {m_centers[i]}: {Ea_m[i]}")


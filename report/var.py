from scipy.stats import norm

from report.config import *

close()
figure(figsize=figsize)
x = np.linspace(-0.05, 0.05, 1000)
n = norm(0.01, 0.01)
y = n.pdf(x)
xlim(-0.03, 0.03)

plot(x, y)

beta = 0.95
var = n.ppf(1 - beta)
idx = x < var
fill_between(x[idx], np.zeros(idx.sum()), y[idx], color='red')
annotate(r'$1 - \beta = 5 \%$', xy=(-0.008, 3), xytext=(-0.02, 20),
         arrowprops=dict(facecolor='black', shrink=0.05))
annotate(r'$ - \text{VaR}_\beta$', xy=(var, 0), xytext=(0.01, 10),
         arrowprops=dict(facecolor='black', shrink=0.05))

axvline(x=0, color='black', lw=1)
xlabel('Return')
ylabel('Density of probability')
gca().xaxis.set_major_formatter(FuncFormatter(lambda p, _: '{:.0f} \%'.format(p*100)))
tight_layout()

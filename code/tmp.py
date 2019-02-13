
#dill.load_session("1_explore.pkl")

import rpy2
print(rpy2.__version__)

import os
os.environ['R_USER'] = './env/Lib/site-packages/rpy2' #path depends on where you installed Python. Mine is the Anaconda distribution

from rpy2.robjects import r
print(r)

r('.libPaths("C:/Users/Uwe/Documents/R/win-library/3.3")')
r('.libPaths()')

from rpy2.robjects.packages import importr
gr = importr('grDevices')
#importr("ggplot2")

r.pi
r.plot(1,1)
gr.dev_off()
r("plot(mtcars$cyl, mtcars$gear)")
gr.dev_off()


import rpy2.robjects.lib.ggplot2 as ggplot2
p = r("p=ggplot(mtcars) + aes(cyl,gear) + geom_line();print(p)")
p.plot()
r('dev.off()')

boxcore = importr("BoxCore")
#ggplot2 = importr('ggplot2')
from rpy2.robjects.vectors import IntVector, FloatVector

blub = boxcore.plot_distr(FloatVector(df.age.values), FloatVector(df.fare.values))
blub.plot()
gr.dev_off()


import rpy2.robjects.pandas2ri


# Marketing-Optimization-in-Python
This code demonstrates marketing optimization in Python where prospects are assigned to offers to maximize total expected responses subject to contraints.
The constraints depicted here are an overall dollar budget upper bound combined with a lower bound on average responder risk score, plus the constraint that
a prospect can receive at most one offer.  The number of prospects is normally quite large, in the millions, whereas the number of alternative offers is usually
quite small, here three.  For illustration, we simulate risk scores from a beta distribution designed to reflect population statistics although actual prospect
risk scores would be employed in reality.  Moreover we generate prospect offer response probabilities directly from risk scores although they would be modeled
from a wider variety of attributes in reality.  Nevertheless, these simulated risk and response scores are sufficient to demo our algorithmic approach.

Our algorithmic approach is to formulate the marketing optimization problem as a large linear binary optimization problem, relax it to a large linear program,
formulate the resulting large, but simple, dual linear program, and then collapse that dual linear program into a nonlinear form with only two nonnegative
variables which is readily solved by a conventional solver.  The quasi-optimal dual results can then be employed to approximately solve the original primal
binary problem.  This approach is both precise and time efficient.  Precision can be gauged by comparing the primal and dual optimal values.  Our demo was run
under Anaconda Navigator, using the Spyder IDE, on a 64-bit PC with four 2.7 GHz processors and 8 GB of RAM.

We ran a SAS production version of this code with simple offer quantity constraints across three products back in the 2000s for a large financial services firm
(see communities.sas.com/thread/4320).  I subsequently developed an R version which is available at raagnew.com/free-trade-fallacy-marketing-optimization-two
-envelope-paradox-congressional-apportionment.html, also at github.com/raagnew/r_utility_code.  This R-code was described in r-bloggers.com/marketing-optimization
-using-the-nonlinear-minimization-function-nlm.  In a different setting, I employed the collapsed dual approach in github.com/raagnew/Constrained-K-Means-Clustering
-in-R.

Bob Agnew (raagnew1@gmail.com, raagnew.com)

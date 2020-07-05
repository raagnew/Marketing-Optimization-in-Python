def mktg_opt(u):
    # Marketing Optimization in Python
    # Bob Agnew, raagnew1@gmail.com, raagnew.com
    from time import time
    import numpy as np
    import pandas as pd
    print()
    print()
    print('Scenario # '+str(u[0]))
    start_time = time()
    np.random.seed(seed=2) # Set seed for repeatable results
    # Risk scores range 300-850.  Higher is less risky.
    # These are Beta distribution simulations, not actual scores.
    # Ref: fico.com/blogs/average-u-s-fico-score-ticks-706
    n = u[1] # Pre-Trim Number of Prospects
    prospect0 = 1 + np.arange(n)
    risk0 = np.round(300+550*np.random.beta(2.57956,.91493,size=n))
    # Trim all prospects with risk scores below 580
    prospect = prospect0[risk0 >= 580]
    risk = risk0[risk0 >= 580]
    x1 = {'Pre-Trim':pd.Series(risk0),
          'Post-Trim':pd.Series(risk)}
    pd.options.display.float_format = '{:,.1f}'.format
    df1 = pd.DataFrame(data=x1)
    print()
    print()
    print("   Risk Score Distribution")
    print (df1.describe())
    
    # Probability response scores for three different credit card offers.
    # These are simulations based on simple cubic risk score interpolations.
    # Riskier prospects are more likely to respond.
    # For given risk score, response is aligned to offer cost.
    # Actual offer response scores would be modeled on various prospect attributes.
    prob1 = .001 + (.3 - .001)*((850 - risk)/550)**3
    prob2 = .0001 + (.2 - .0001)*((850 - risk)/550)**3
    prob3 = .00001 + (.1 - .00001)*((850 - risk)/550)**3
    # Offer Unit Costs
    cost = [.50,.25,.10]
    # Budget Dollar Upper Bound
    budget = float(u[2])
    # Average Risk Score Lower Bound
    avg_risk = float(u[3])
    x2 = {'Measure':['Offer Dollar Budget','Average Risk Score'],
          'Bound':[budget,avg_risk]}
    df2 = pd.DataFrame(data=x2)
    print()
    print()
    print('      Stipulated Constraints')
    print(df2.to_string(index=False,justify='right'))
    
    # Dual Optimization
    n = np.size(prospect)
    z = np.zeros(n)
    v = np.full(n,avg_risk)
    def dual(y):
        d1 = prob1 - cost[0]*y[0] - prob1*(v - risk)*y[1]
        d2 = prob2 - cost[1]*y[0] - prob2*(v - risk)*y[1]
        d3 = prob3 - cost[2]*y[0] - prob3*(v - risk)*y[1]
        d = np.array([z,d1,d2,d3])
        val = budget*y[0] + sum(np.amax(d,axis=0))
        return val
    from scipy.optimize import minimize
    bnds = ((0,None),(0,None))
    res = minimize(dual,(0,0),method='L-BFGS-B',bounds=bnds)
    print()
    print()
    print('                    Quasi-Optimal Dual Solution')
    txt = 'Minimum Dual Value = {xxx:,.1f}'+' (Compare to Total Expected Responses)'
    print(txt.format(xxx = res.fun))
    print('Minimum Dual Parameters = '+str(res.x))
    y = res.x
    
    # Primal Assignments
    d1 = prob1 - cost[0]*y[0] - prob1*(v - risk)*y[1]
    d2 = prob2 - cost[1]*y[0] - prob2*(v - risk)*y[1]
    d3 = prob3 - cost[2]*y[0] - prob3*(v - risk)*y[1]
    d = np.array([d1,d2,d3])
    w = np.amax(d,axis=0)
    offer = 1 + np.argmax(d,axis=0)
    order = np.flip(np.argsort(w)) # Offers in descending order
    w = w[order]
    prospect = prospect[order]
    risk = risk[order]
    offer = offer[order]
    prob1 = prob1[order]
    prob2 = prob2[order]
    prob3 = prob3[order]
    c = cost[0]*(offer==1) + cost[1]*(offer==2) + cost[2]*(offer==3)
    cumcost = np.cumsum(c)
    offer[np.logical_or(w < 0,cumcost > budget)] = 0
    offers1 = float(sum(offer==1))
    offers2 = float(sum(offer==2))
    offers3 = float(sum(offer==3))
    total_offers = offers1 + offers2 + offers3
    resp1 = sum(prob1[offer==1])
    resp2 = sum(prob2[offer==2])
    resp3 = sum(prob3[offer==3])
    total_resp = resp1 + resp2 + resp3
    cost1 = cost[0]*offers1
    cost2 = cost[1]*offers2
    cost3 = cost[2]*offers3
    total_cost = cost1 + cost2 + cost3
    risk1 = sum(prob1[offer==1]*risk[offer==1])/(resp1 + 1e-15)
    risk2 = sum(prob2[offer==2]*risk[offer==2])/(resp2 + 1e-15)
    risk3 = sum(prob3[offer==3]*risk[offer==3])/(resp3 + 1e-15)
    average_risk = (resp1*risk1 + resp2*risk2 + resp3*risk3)/total_resp
    x3 = {'Offer':['# 1','# 2','# 3','Total'],
               'Quantity':[offers1,offers2,offers3,total_offers],
               'Total $ Cost':[cost1,cost2,cost3,total_cost],
               'Expected Responses':[resp1,resp2,resp3,total_resp],
               'Avg Responder Risk Score':[risk1,risk2,risk3,average_risk]}
    df3 = pd.DataFrame(data=x3)
    print()
    print()
    print('                         Quasi-Optimal Primal Solution')
    print(df3.to_string(index=False,justify='right'))
    print()
    print()
    txt = 'Cost Per Response = ${xxx:,.1f}'
    print(txt.format(xxx = total_cost/total_resp))
    print()
    print()
    offered = np.arange(int(total_offers))
    prospect = prospect[offered]
    risk = risk[offered]
    offer = offer[offered]
    order = np.argsort(prospect,axis=0)
    prospect = prospect[order]
    risk = risk[order]
    offer = offer[order]
    x4 = {'Prospect':prospect,'Risk':risk,'Offer':offer}
    df4 = pd.DataFrame(data=x4)
    print('First 50 Sorted Prospect Offers')
    pd.options.display.float_format = '{:,.0f}'.format
    print(df4.head(50).to_string(index=False,justify='right'))
    print()
    print()
    end_time = time()
    print('Elapsed Seconds = '+str(end_time - start_time))
    print()
    print()
    # To save entire campaign list:
    # import csv
    # df4.to_csv('c:/Marketing Optimization/Campaign_List.csv')
    return  


    
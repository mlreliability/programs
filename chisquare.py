#Calculate Chi-square statistic of each feature using categorical feature set
import pandas as pd
from scipy.stats import chi2_contingency


df= pd.read_csv('dataset.csv', encoding= 'unicode_escape')
c=0
categories = df[df.columns[559:]]
features = df[df.columns[:558]]
for feature in features:
    for category in categories:
        # Create a contingency table
        contingency_table = pd.crosstab(df[feature], df[category])

        # Perform Chi-Square test
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        if p < 0.05:
          print(f"The correlation between {feature} is {chi2} with a p-value of {p}and degree of freedom is {dof}")
          c=c+1
print(c)
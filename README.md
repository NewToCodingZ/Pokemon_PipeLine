OverView-- This project uses the PokéAPI to collect and analyze data on the first 310 Pokémon, transforming and visualizing it with Python libraries like pandas, matplotlib, numpy, seaborn, scipy, and sklearn. 
The processed data is stored and queried using SQLite, showcasing skills in ETL (Extract, Transform, Load), data visualization, machine learning, and SQL database management.

Technologies Used
pandas – Data cleaning and manipulation
numpy – Numerical operations
matplotlib & seaborn – Data visualization
scikit-learn – Dimensionality reduction (t-SNE) & regression
scipy – Statistical analysis
sqlite3 – Database creation and querying
requests / json – API interaction

Data Collection
Used the PokéAPI to retrieve detailed information on Pokémon (name, types, stats, abilities, etc.)
Wrote a script using requests to fetch and parse JSON data
Stored data into a pandas DataFrame

Data Wrangling & Transformation
Cleaned and standardized features such as:
Base stats (HP, Attack, Defense, Speed)
Primary and Secondary Types
Evolution chain
Generations

📊 Data Visualizations
Basic Charts
Bar Charts – Count of Pokémon per Type
Line Charts – Evolution levels vs base stats
Bubble Charts – HP vs Speed with size = Weight

Statistical & Exploratory Charts
Scatter Plots – Attack vs Defense
Heatmaps – Correlation between stats
Regression Plots – Stats predictions using seaborn.regplot
Radar Charts – Stat profiles per Pokémon
t-SNE (t-distributed stochastic neighbor embedding) – High-dimensional clustering of Pokémon using base stats (sklearn.manifold.TSNE)
Used scikit-learn for dimensionality reduction (t-SNE)
Performed linear regression to model relationships between stats (e.g., Predict HP from Attack and Defense)
Used scipy.stats to run statistical tests like correlation coefficients and distribution checks

🗄️ SQLite Integration
Table Creation
Pokemon – stores main Pokémon data
Types – stores Type information
Evolutions – evolutionary paths
Generations – region and generation mapping
Used JOINS to combined the charts
Used WHERE, AND, ORDER BY, GROUP BY, HAVING, and others. 






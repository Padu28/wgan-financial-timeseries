# Publishing on GitHub

1. Create an empty public repository named `wgan-financial-timeseries` at https://github.com/new (no README, licence or .gitignore).
2. Turn on Pages first: *Settings > Pages > Source: GitHub Actions*.
3. From the project folder (the one containing `README.md`):

   ```bash
   git init
   git config user.name "Padmashree Shah"
   git config user.email "Padmashreeshah@gmail.com"
   git add .
   git commit -m "Tempora v2.0.0: synthetic financial time series with WGAN-GP"
   git branch -M main
   git remote add origin https://github.com/Padu28/wgan-financial-timeseries.git
   git push -u origin main
   ```

4. Both workflows should turn green under *Actions*; the dashboard is then live at https://padu28.github.io/wgan-financial-timeseries/
5. Add topics such as `gan`, `wgan-gp`, `synthetic-data`, `time-series`, `quantitative-finance`, `risk-management`, `deep-learning`, `numpy`, `pytorch`, `data-visualization`.

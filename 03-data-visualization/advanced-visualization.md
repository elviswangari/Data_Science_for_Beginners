# Advanced Data Visualization Techniques

[⬅️ Previous: Matplotlib and Seaborn](matplotlib-seaborn.md) | [Next: ML Fundamentals ➡️](../04-machine-learning/ml-fundamentals.md)

## Learning Objectives

By the end of this section, you will:

1. Master interactive visualizations with Plotly
2. Create advanced statistical visualizations
3. Learn geographic data visualization
4. Understand 3D plotting techniques
5. Build interactive dashboards

## 1. Interactive Visualization with Plotly

### Basic Plotly Usage

```python
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'x': np.linspace(0, 10, 100),
    'y': np.sin(np.linspace(0, 10, 100)),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Create interactive line plot
fig = px.line(df, x='x', y='y', title='Interactive Line Plot')
fig.show()
```

### Advanced Plotly Features

```python
# Create scatter plot with multiple features
fig = px.scatter(df, x='x', y='y',
                 color='category',
                 size=np.abs(df['y']),
                 hover_data=['category'],
                 title='Interactive Scatter Plot')

# Customize layout
fig.update_layout(
    plot_bgcolor='white',
    showlegend=True,
    hovermode='closest'
)

fig.show()
```

## 2. Statistical Visualization

### Advanced Box Plots

```python
# Create advanced box plot with points
fig = go.Figure()

for category in df['category'].unique():
    subset = df[df['category'] == category]
    
    fig.add_trace(go.Box(
        y=subset['y'],
        name=category,
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))

fig.update_layout(title='Advanced Box Plot with Points')
fig.show()
```

### Violin Plots with Split

```python
# Create split violin plot
fig = go.Figure()

for category in df['category'].unique():
    subset = df[df['category'] == category]
    
    fig.add_trace(go.Violin(
        x=subset['category'],
        y=subset['y'],
        name=category,
        side='positive',
        points='all'
    ))

fig.update_layout(title='Split Violin Plot')
fig.show()
```

## 3. Geographic Visualization

### Basic Maps with Plotly

```python
# Create sample geographic data
geo_data = pd.DataFrame({
    'lat': np.random.uniform(20, 60, 100),
    'lon': np.random.uniform(-120, -60, 100),
    'value': np.random.normal(0, 1, 100)
})

# Create scatter mapbox
fig = px.scatter_mapbox(geo_data,
                        lat='lat',
                        lon='lon',
                        color='value',
                        size=np.abs(geo_data['value']),
                        title='Geographic Data Visualization')

fig.update_layout(mapbox_style='open-street-map')
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()
```

### Choropleth Maps

```python
# Example with US states data
import json

# Create sample state data
states_data = pd.DataFrame({
    'state': ['CA', 'NY', 'TX', 'FL'],
    'value': [1, 2, 3, 4]
})

fig = px.choropleth(states_data,
                    locations='state',
                    locationmode='USA-states',
                    color='value',
                    scope='usa',
                    title='US States Choropleth Map')
fig.show()
```

## 4. 3D Visualization

### 3D Scatter Plots

```python
# Create 3D data
df_3d = pd.DataFrame({
    'x': np.random.normal(0, 1, 100),
    'y': np.random.normal(0, 1, 100),
    'z': np.random.normal(0, 1, 100),
    'color': np.random.uniform(0, 1, 100)
})

# Create 3D scatter plot
fig = px.scatter_3d(df_3d, x='x', y='y', z='z',
                    color='color',
                    title='3D Scatter Plot')
fig.show()
```

### 3D Surface Plots

```python
# Create surface data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create 3D surface plot
fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
fig.update_layout(title='3D Surface Plot')
fig.show()
```

## 5. Interactive Dashboards

### Creating a Dashboard with Plotly

```python
from plotly.subplots import make_subplots

def create_interactive_dashboard(df):
    """Create an interactive dashboard with multiple plots"""
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Scatter Plot', 'Line Plot',
                       'Box Plot', 'Histogram')
    )
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=df['x'], y=df['y'], mode='markers'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['x'], y=df['y'], mode='lines'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(y=df['y']),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=df['y']),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=800, showlegend=False,
                     title_text="Interactive Dashboard")
    fig.show()
```

## 6. Custom Visualization Functions

### Time Series Animation

```python
def create_animated_timeseries(df, date_column, value_column):
    """Create an animated time series plot"""
    
    fig = px.scatter(df, x=date_column, y=value_column,
                     animation_frame=date_column,
                     animation_group=value_column,
                     range_y=[df[value_column].min(),
                             df[value_column].max()])
    
    fig.update_layout(
        title='Animated Time Series',
        xaxis_title=date_column,
        yaxis_title=value_column
    )
    
    return fig
```

### Custom Color Scales

```python
def create_custom_heatmap(data, custom_colorscale):
    """Create a heatmap with custom color scale"""
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=custom_colorscale,
        showscale=True
    ))
    
    fig.update_layout(
        title='Custom Colored Heatmap',
        xaxis_title='X Axis',
        yaxis_title='Y Axis'
    )
    
    return fig
```

## 7. Best Practices for Interactive Visualization

1. **Performance Optimization**

```python
def optimize_large_dataset(df, max_points=1000):
    """Optimize large dataset for visualization"""
    
    if len(df) > max_points:
        # Randomly sample data
        df = df.sample(n=max_points, random_state=42)
    
    return df
```

2. **Responsive Layout**

```python
def create_responsive_layout(fig):
    """Make plot layout responsive"""
    
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    return fig
```

## 8. Practice Exercise

Create an interactive dashboard for data analysis:

```python
def create_analysis_dashboard(df):
    """Create comprehensive analysis dashboard"""
    
    # Optimize data if needed
    df = optimize_large_dataset(df)
    
    # Create dashboard
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "box"}, {"type": "heatmap"}],
               [{"type": "scatter3d", "colspan": 2}, None]],
        subplot_titles=('Time Series', 'Distribution',
                       'Box Plot', 'Correlation',
                       '3D Visualization')
    )
    
    # Add plots
    fig.add_trace(
        go.Scatter(x=df.index, y=df['value'], mode='lines+markers'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=df['value']),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(y=df['value'], quartilemethod="linear"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Heatmap(z=df.corr()),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter3d(x=df['x'], y=df['y'], z=df['z'],
                     mode='markers'),
        row=3, col=1
    )
    
    # Update layout
    fig = create_responsive_layout(fig)
    fig.update_layout(height=1000, showlegend=False)
    
    return fig
```

---

## Navigation

[⬅️ Previous: Matplotlib and Seaborn](matplotlib-seaborn.md) | [Next: ML Fundamentals ➡️](../04-machine-learning/ml-fundamentals.md)

[🔝 Back to Top](#advanced-data-visualization-techniques)

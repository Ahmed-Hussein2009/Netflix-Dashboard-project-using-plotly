import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from textblob import TextBlob
import plotly.io as plt_io

app = dash.Dash(__name__, title="Netflix Dashboard ")
netflix_df = pd.read_csv('data/netflix_titles.csv')


def clean_netflix_df(df):
    df['country'] = df['country'].fillna(df['country'].mode()[0])
    df['cast'].replace(np.nan, 'No Data', inplace=True)
    df['director'].replace(np.nan, 'No Data', inplace=True)
    df.dropna(inplace=True)

    df.drop_duplicates(inplace=True)

    df["date_added"] = pd.to_datetime(df['date_added'])
    df['month_added'] = df['date_added'].dt.month
    df['month_name_added'] = df['date_added'].dt.month_name()
    df['year_added'] = df['date_added'].dt.year

    df['first_country'] = df['country'].apply(lambda x: x.split(",")[0])
    df['first_country'].replace('United States', 'USA', inplace=True)
    df['first_country'].replace('United Kingdom', 'UK', inplace=True)
    df['first_country'].replace('South Korea', 'S. Korea', inplace=True)

    netflix_df['count'] = 1
    df['genre'] = df['listed_in'].apply(lambda x: x.replace(' ,', ',').replace(', ', ',').split(','))
    return df
netflix_df = clean_netflix_df(netflix_df)

# netflix_df=netflix_df.drop_duplicates()

def fig_bar_horiz():
    country_order = netflix_df['first_country'].value_counts()[:11].index
    data_q2q3 = netflix_df[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[
        country_order]
    data_q2q3['sum'] = data_q2q3.sum(axis=1)
    data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']].sort_values(by='Movie', ascending=False)[
                      ::-1]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=data_q2q3_ratio.index,
        x=round(data_q2q3_ratio['Movie'] * 100, 2),
        name='Movies',
        orientation='h',

        marker=dict(
            color='#b20710',
            line=dict( width=3),
        )
    ))
    fig_bar.add_trace(go.Bar(
        y=data_q2q3_ratio.index,
        x=round(data_q2q3_ratio['TV Show'] * 100, 2),
        name='TV Shows',
        orientation='h',
        marker=dict(
            color='#221f1f',
            line=dict(width=3)
        )
    ))

    fig_bar.update_layout(barmode='stack',
                          title={
                              'text': "Top 10 countries Movie & TV Show split",
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'}
                          )
    return fig_bar

def fig_bar_stacked():
    order = pd.DataFrame(netflix_df.groupby('rating')['count'].sum().sort_values(ascending=False).reset_index())
    rating_order = list(order['rating'])
    mf = netflix_df.groupby('type')['rating'].value_counts().unstack().sort_index().fillna(0).astype(int)[rating_order]

    movie = mf.loc['Movie']
    tv = - mf.loc['TV Show']

    fig_stacked = go.Figure()
    fig_stacked.add_trace(go.Bar(x=movie.index, y=movie, name='Movies',marker_color='#b20710'))
    fig_stacked.add_trace(go.Bar(x=tv.index, y=tv, name='TVShows',marker_color='#221f1f'))
    fig_stacked.update_layout(barmode='relative',
                              title={
                                      'text': 'Rating distribution by Movie & TV Show',
                                       'y': 0.9,
                                       'x': 0.5,
                                       'xanchor': 'center',
                                       'yanchor': 'top'}
                             )
    return fig_stacked

def fig_stack_without_flying():
    dfx = netflix_df[['release_year', 'description']]
    dfx = dfx.rename(columns={'release_year': 'Release Year'})
    for index, row in dfx.iterrows():
        z = row['description']
        testimonial = TextBlob(z)
        p = testimonial.sentiment.polarity
        if p == 0:
            sent = 'Normal'
        elif p > 0:
            sent = 'Positive'
        else:
            sent = 'Negative'
        dfx.loc[[index, 2], 'Sentiment'] = sent

    dfx = dfx.groupby(['Release Year', 'Sentiment']).size().reset_index(name='Total Content')

    dfx = dfx[dfx['Release Year'] >= 2010]
    fig_stacked_without_fly = px.bar(dfx, x="Release Year", y="Total Content",color='Sentiment')
    fig_stacked_without_fly.update_layout(title={
        'text': 'Sentiment Analysis over years for Movies and Tv Shows',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    return fig_stacked_without_fly

def fig_pie_purst():
    fig_purst = px.sunburst(netflix_df[netflix_df['year_added'] >= 2018], path=['year_added', 'month_name_added'],
                            values='count', color_continuous_scale='armyrose')
    fig_purst.update_layout(title={
        'text': 'Number of Movies and Tv shows added per month last 5 year',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    return fig_purst




############################################################################################################
# def gen_comp():
#     # For viz: Ratio of Movies & TV shows
#     x=netflix_df.groupby(['type'])['type'].count()
#     y=len(netflix_df)
#     r=((x/y)).round(2)
#     mf_ratio = pd.DataFrame(r).T
    
#     top_labels = ['Movie', 'TV Show']
#     colors = ['#b20710', '#221f1f']
#     x_data = [[int(mf_ratio['Movie']*100), int(mf_ratio['TV Show']*100)]]
#     y_data = ['Movie & TV Show distribution']
#     fig = go.Figure()

#     for i in range(0, len(x_data[0])):
#         for xd, yd in zip(x_data, y_data):
#             fig.add_trace(go.Bar(
#                 x=[xd[i]], y=[yd],
#                 orientation='h',
#                 marker=dict(
#                     color=colors[i],
#                     line=dict(color='rgb(248, 248, 249)', width=1)
#                 )
#             ))

#     fig.update_layout(
#         xaxis=dict(
#             showgrid=False,
#             showline=False,
#             showticklabels=False,
#             zeroline=False,
#             domain=[0.15, 1]
#         ),
#         yaxis=dict(
#             showgrid=False,
#             showline=False,
#             showticklabels=False,
#             zeroline=False,
#         ),
#         barmode='stack',
#         paper_bgcolor='white',
#         plot_bgcolor='white',
#         margin=dict(l=140, r=10, t=90, b=80),
#         showlegend=False,
#     )

#     annotations = []

#     for yd, xd in zip(y_data, x_data):
#         # labeling the y-axis
#         annotations.append(dict(xref='paper', yref='y',
#                                 x=0.14, y=yd,
#                                 xanchor='right',
#                                 text=str(yd),
#                                 font=dict(family='serif', size=20,
#                                           color='rgb(67, 67, 67)'),
#                                 showarrow=False, align='right'))
#         # labeling the first percentage of each bar (x_axis)
#         annotations.append(dict(xref='x', yref='y',
#                                 x=xd[0] / 2, y=yd,
#                                 text=str(xd[0]) + '%',
#                                 font=dict(family='serif', size=40,
#                                           color='white'),
#                                 showarrow=False))
#         # labeling the first Likert scale (on the top)
#         if yd == y_data[-1]:
#             annotations.append(dict(xref='x', yref='paper',
#                                     x=xd[0] / 2, y=1.1,
#                                     text=top_labels[0],
#                                     font=dict(family='serif', size=40,
#                                               color='rgb(67, 67, 67)'),
#                                     showarrow=False))
#         space = xd[0]
#         for i in range(1, len(xd)):
#                 # labeling the rest of percentages for each bar (x_axis)
#                 annotations.append(dict(xref='x', yref='y',
#                                         x=space + (xd[i]/2), y=yd,
#                                         text=str(xd[i]) + '%',
#                                         font=dict(family='serif', size=40,
#                                                   color='white'),
#                                         showarrow=False))
#                 # labeling the Likert scale
#                 if yd == y_data[-1]:
#                     annotations.append(dict(xref='x', yref='paper',
#                                             x=space + (xd[i]/2), y=1.1,
#                                             text=top_labels[i],
#                                             font=dict(family='serif', size=40,
#                                                       color='rgb(67, 67, 67)'),
#                                             showarrow=False))
#                 space += xd[i]

#     return fig.update_layout(annotations=annotations)




############################################################################################################


app.layout = html.Div(children=[

    html.H1(id='header', children=[html.Div("Netflix Movies and TvShows Analysis", id='header-text')],
            style={'textAlign': 'center', 'color': '#b20710'}, className="mb-3"),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(children=[
        html.Div(children=[
            html.P('Filter by Release Year', className='fix_label', style={'text-align': 'center', 'color': '#221f1f'}),
            dcc.Slider(id='slider_year',
                       included=True,
                       updatemode='drag',
                       tooltip={'always_visible': True},
                       min=1925,
                       max=2020,
                       step=1,
                       value=2009,
                       marks={str(yr): str(yr) for yr in range(1925, 2020, 10)}, className="row"
                       ),
            html.Div([
                html.Div([
                    dcc.Graph(id='line_chart', config={'displayModeBar': 'hover'}), ], className="row",
                    style={'width': '100%'}),

            ]),
        ], className="col-md-6"),
        html.Div(children=[
            html.Div([
                dcc.Graph(id='bar_fig', figure=fig_bar_horiz())], id='FigBarGraphDiv')
        ], className="col-md-6"),

    ], className="row"),
    html.Div(children=[
        html.Div(children=[
            html.Div([
                dcc.Graph(id='stack_fig', figure=fig_bar_stacked())], id='StackedGraphDiv')

        ], className="col-md-6"),
        html.Div(children=[
            html.Div([
                dcc.Graph(id='SentBarGraphDiv', figure=fig_stack_without_flying()), ])
        ], className="col-md-6"),

    ], className="row"),

    html.Div(children=[
        html.Div(children=[
            html.Div([
                dcc.Graph(id='purst_fig', figure=fig_pie_purst())],
                id='PurstGraphDiv')

        ], className="col-md-6"),
        html.Div(children=[
            html.Label('Movie Statistics Calculator', id='calculator'),
            html.Div([
                dcc.Dropdown(id='dropDown', options=[{'label': x, 'value': x} for x in netflix_df['first_country'].unique()],
                             value='Egypt'),
                html.Br(),
                html.Br(),
                html.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td("No. of Movies till date"),

                            html.Td([
                                html.Div(
                                    id="val1"

                                )

                            ])
                        ]),
                        html.Tr([
                            html.Td("No. of TV Shows till date"),

                            html.Td([
                                html.Div(
                                    id="val2"

                                )

                            ])
                        ]),
                        html.Tr([
                            html.Td("Top Actor"),

                            html.Td([
                                html.Div(
                                    id="val3"

                                )

                            ])
                        ]),
                        html.Tr([
                            html.Td("Top Director"),

                            html.Td([
                                html.Div(
                                    id="val4"

                                )

                            ])
                        ])

                    ])
                ], className="table table-striped")
            ])
        ], className="col-md-6"),

    ], className="row"),

], className="container-fluid",)


@app.callback(
    [Output('val1', 'children'), Output('val2', 'children'), Output('val3', 'children'), Output('val4', 'children')],
    Input('dropDown', 'value')
)
def updateTable(dropDown):
    dfx = netflix_df[['type', 'country']]
    dfMovie = dfx[dfx['type'] == 'Movie']
    dfTV = dfx[dfx['type'] == 'TV Show']
    dfM1 = dfMovie['country'].str.split(',', expand=True).stack()
    dfTV1 = dfTV['country'].str.split(',', expand=True).stack()
    dfM1 = dfM1.to_frame()
    dfTV1 = dfTV1.to_frame()
    dfM1.columns = ['country']
    dfTV1.columns = ['country']
    dfM2 = dfM1.groupby(['country']).size().reset_index(name='counts')
    dfTV2 = dfTV1.groupby(['country']).size().reset_index(name='counts')
    dfM2['country'] = dfM2['country'].str.strip()
    dfTV2['country'] = dfTV2['country'].str.strip()
    val11 = dfM2[dfM2['country'] == dropDown]
    val22 = dfTV2[dfTV2['country'] == dropDown]
    val11 = val11.reset_index()
    val22 = val22.reset_index()

    if val11.empty:
        val1 = 0
    else:
        val1 = val11.loc[0]['counts']

    if val22.empty:
        val2 = 0
    else:
        val2 = val22.loc[0]['counts']

    # Top Actor
    dfA = netflix_df[['cast', 'country']]
    dfA1 = dfA[dfA['country'].str.contains(dropDown, case=False)]
    dfA2 = dfA1['cast'].str.split(',', expand=True).stack()
    dfA2 = dfA2.to_frame()
    dfA2.columns = ['Cast']
    dfA3 = dfA2.groupby(['Cast']).size().reset_index(name='counts')
    dfA3 = dfA3[dfA3['Cast'] != 'No Cast Specified']
    dfA3 = dfA3.sort_values(by='counts', ascending=False)
    if dfA3.empty:
        val3 = "Actor data from this country is not available"
    else:
        val3 = dfA3.iloc[0]['Cast']
    # Top Director
    dfD = netflix_df[['director', 'country']]
    dfD1 = dfD[dfD['country'].str.contains(dropDown, case=False)]
    dfD2 = dfD1['director'].str.split(',', expand=True).stack()
    dfD2 = dfD2.to_frame()
    dfD2.columns = ['Director']
    dfD3 = dfD2.groupby(['Director']).size().reset_index(name='counts')
    dfD3 = dfD3[dfD3['Director'] != 'No Director Specified']
    dfD3 = dfD3.sort_values(by='counts', ascending=False)
    if dfD3.empty:
        val4 = "Director data from this country is not available"
    else:
        val4 = dfD3.iloc[0]['Director']
    return val1, val2, val3, val4

@app.callback(Output('line_chart', 'figure'),
              [Input('slider_year', 'value')])
def update_graph(slider_year):
    type_movie = netflix_df[(netflix_df['type'] == 'Movie')][['type', 'release_year']]
    type_movie['type1'] = type_movie['type']
    type_movie_1 = type_movie.groupby(['release_year', 'type1'])['type'].count().reset_index()
    filter_movie = type_movie_1[(type_movie_1['release_year'] >= slider_year)]

    type_tvshow = netflix_df[(netflix_df['type'] == 'TV Show')][['type', 'release_year']]
    type_tvshow['type2'] = type_tvshow['type']
    type_tvshow_1 = type_tvshow.groupby(['release_year', 'type2'])['type'].count().reset_index()
    filter_tvshow = type_tvshow_1[(type_tvshow_1['release_year'] >= slider_year)]

    return {
        'data': [go.Scatter(
            x=filter_movie['release_year'],
            y=filter_movie['type'],
            mode='markers+lines',
            name='Movie',
            line=dict(shape="spline", smoothing=1.3, width=3, color='#b20710'),
            marker=dict(size=10, symbol='circle', color='#f5f5f1',
                        line=dict(color='blue', width=2)
                        ),

            hoverinfo='text',
            hovertext=
            '<b>Release Year</b>: ' + filter_movie['release_year'].astype(str) + '<br>' +
            '<b>Type</b>: ' + filter_movie['type1'].astype(str) + '<br>' +
            '<b>Count</b>: ' + [f'{x:,.0f}' for x in filter_movie['type']] + '<br>'
        ),

            go.Scatter(
                x=filter_tvshow['release_year'],
                y=filter_tvshow['type'],
                mode='markers+lines',
                name='TV Show',
                line=dict(shape="spline", smoothing=1.3, width=3,color='#221f1f' ),
                marker=dict(size=10, symbol='circle',color='#f5f5f1',
                            line=dict(color='blue',width=2)
                            ),

                hoverinfo='text',
                hovertext=
                '<b>Release Year</b>: ' + filter_tvshow['release_year'].astype(str) + '<br>' +
                '<b>Type</b>: ' + filter_tvshow['type2'].astype(str) + '<br>' +
                '<b>Count</b>: ' + [f'{x:,.0f}' for x in filter_tvshow['type']] + '<br>'

            )],
        'layout': go.Layout(
            title={
                'text': 'Movies and TV Shows by Release Year',
                'xanchor': 'right',
                'yanchor': 'top'},
            width=1200
        ),

    }


if __name__ == '__main__':
    app.run_server(debug=True)


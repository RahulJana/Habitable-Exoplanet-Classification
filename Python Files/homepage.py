import streamlit as st
from PIL import Image
image = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\habitable.png')


def app():
    st.image(image, caption='habitable zone')
    st.write('Our Solar System formed around 4600 million years ago from a supernova explosion. '
             'In that event, everything is created in our Solar System. '
             'The gravity pulled all the materials into clumps and rounding some of them, '
             'later forming planets, dwarf planets, moons, and all the fundamental components '
             'of the solar system (such as asteroids, rings, meteoroids, comets, Kuiper belt, etc.). '
             )
    image2 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\brightplanet.png')
    st.image(image2, caption='Exoplanet visual')
    st.subheader('But what are Exoplanets?')
    st.write('Exoplanets are planets beyond our own solar system. Thousands have been discovered in the past two decades, mostly with NASA’s Kepler Space Telescope.'
            'These exoplanets come in a huge variety of sizes and orbits. Some are gigantic planets hugging close to their parent stars; others are icy, some rocky. NASA and other agencies are looking for a special kind of planet: one that’s the same size as Earth, orbiting a sun-like star in the habitable zone.'
'The habitable zone is the area around a star where it is not too hot and not too cold for liquid water to exist on the surface of surrounding planets. Imagine if Earth was where Pluto is. The Sun would be barely visible (about the size of a pea) and Earth’s ocean and much of its atmosphere would freeze.')


    st.write('Kepler Space Telescope’s primary method of searching for planets was the “Transit” method.'

    'Transit method: In the diagram below, a star is orbited by a planet. From the graph, it is visible that the starlight intensity drops because it is partially obscured by the planet, given our position. The starlight rises back to its original value once the planets crosses in front of the star. ')
    image3 = Image.open(r'C:\Users\Arnab-PC\PycharmProjects\ExoPlanetFinal\unnamed.png')
    st.image(image3, caption='Transit Method')
    st.subheader('About project')
    st.write('This project works on the parameters that were recorded by the Kepler telescope (NASA) regarding the exoplanets, to classify the planes that are most suitable to live on(habitable) the project will classify based on a large number of parameter. The outline can be described as a classifier that takes a huge amount of data regarding the planets and then finding if it is habitable, which would help in scientific explorations of life forms. ')




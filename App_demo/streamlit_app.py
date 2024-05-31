import streamlit as st
import requests
import zipfile
import io
from utils import icon
from streamlit_image_select import image_select

# UI configurations
st.set_page_config(page_title="Replicate Image Generator",
                   page_icon=":bridge_at_night:",
                   layout="wide")
icon.show_icon(":foggy:")
st.markdown("# :rainbow[Text-to-Image Artistry Studio]")


# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()


def configure_sidebar() -> None:
    """
    Setup and display the sidebar elements.

    This function configures the sidebar of the Streamlit application, 
    including the form for user inputs and the resources section.
    """
    with st.sidebar:
        with st.form("my_form"):
            st.info("**Start here ↓**", icon="👋🏾")
            prompt = st.text_area(
                ":orange[**Enter prompt: Typing your prompt✍🏾**]",
                value="An astronaut riding a rainbow unicorn, cinematic, dramatic",
                height=200)
            negative_prompt = st.text_area(":orange[**Negative prompt 🙅🏽‍♂️**]",
                                           value="the absolute worst quality, distorted features",
                                           help="This is a negative prompt, basically type what you don't want to see in the generated image",
                                           height=150)
            
            num_outputs = st.slider(
                "Number of images to output", value=1, min_value=1, max_value=4)
            width = st.number_input("Width of output image", value=1024)
            height = st.number_input("Height of output image", value=1024)

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True)

        return submitted, width, height, num_outputs, prompt, negative_prompt


def main_page(submitted: bool, width: int, height: int, num_outputs: int,
              prompt: str, negative_prompt: str) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
        width (int): Width of the output image.
        height (int): Height of the output image.
        num_outputs (int): Number of images to output.
        prompt (str): Text prompt for the image generation.
        negative_prompt (str): Text prompt for elements to avoid in the image.
    """
    if submitted:
        with st.status('👩🏾‍🍳 Whipping up your words into art...', expanded=True) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and strecth in the meantime")
            try:
                # Only call the API if the "Submit" button was pressed
                if submitted:
                    # Calling the replicate API to get the image
                    with generated_images_placeholder.container():
                        all_images = []  # List to store all generated images
                        output = 
                        if output:
                            st.toast(
                                'Your image has been generated!', icon='😍')
                            # Save generated image to session state
                            st.session_state.generated_image = output

                            # Displaying the image
                            for image in st.session_state.generated_image:
                                with st.container():
                                    st.image(image, caption="Generated Image 🎈",
                                             use_column_width=True)
                                    # Add image to the list
                                    all_images.append(image)

                                    response = requests.get(image)
                        # Save all generated images to session state
                        st.session_state.all_images = all_images

            except Exception as e:
                print(e)
                st.error(f'Encountered an error: {e}', icon="🚨")
        pass
    # If not submitted, chill here 🍹
    else:
        pass

    # Gallery display for inspo
    with gallery_placeholder.container():
        img = image_select(
            label="Like what you see? Right-click and save! It's not stealing if we're sharing! 😉",
            images=[
                "gallery/farmer_sunset.png", "gallery/astro_on_unicorn.png",
                "gallery/friends.png", "gallery/wizard.png", "gallery/puppy.png",
                "gallery/cheetah.png", "gallery/viking.png",
            ],
            captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
                      "An astronaut riding a rainbow unicorn, cinematic, dramatic",
                      "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
                      "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
                      "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
                      "A cheetah mother nurses her cubs in the tall grass of the Serengeti. The early morning sun beams down through the grass. National Geographic photography by Frans Lanting",
                      "A close-up portrait of a bearded viking warrior in a horned helmet. He stares intensely into the distance while holding a battle axe. Dramatic mood lighting, digital oil painting",
                      ],
            use_container_width=True
        )


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the sidebar configuration and the main page layout.
    It retrieves the user inputs from the sidebar, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    submitted, width, height, num_outputs, prompt, negative_prompt = configure_sidebar()
    main_page(submitted, width, height, num_outputs, prompt, negative_prompt)


if __name__ == "__main__":
    main()

import streamlit as st
from PIL import Image
from utils import icon
from streamlit_image_select import image_select
from streamlit_drawable_canvas import st_canvas
import random

import sys
sys.path.append('..')
from base_gen import load_model_base, gen_base
from image_condition_gen import load_controlnet_model, gen_controlnet
from inpainting_gen import load_model_inpaint, inpaint_gen
from utils_func import translate_to_eng


#pipeline
pipe = None

# UI configurations
st.set_page_config(page_title="Interor and Architect Generator",
                   page_icon=":bridge_at_night:",
                   layout="wide")
icon.show_icon(":foggy:")
st.markdown("# :rainbow[Interior and Architerct Generation]")


# Placeholders for images and gallery
generated_images_placeholder = st.empty()
gallery_placeholder = st.empty()


    # Gallery display for inspo
    # with gallery_placeholder.container():
    #     img = image_select(
    #         label="Like what you see? Right-click and save! It's not stealing if we're sharing! 😉",
    #         images=[
    #             "gallery/farmer_sunset.png", "gallery/astro_on_unicorn.png",
    #             "gallery/friends.png", "gallery/wizard.png", "gallery/puppy.png",
    #             "gallery/cheetah.png", "gallery/viking.png",
    #         ],
    #         captions=["A farmer tilling a farm with a tractor during sunset, cinematic, dramatic",
    #                   "An astronaut riding a rainbow unicorn, cinematic, dramatic",
    #                   "A group of friends laughing and dancing at a music festival, joyful atmosphere, 35mm film photography",
    #                   "A wizard casting a spell, intense magical energy glowing from his hands, extremely detailed fantasy illustration",
    #                   "A cute puppy playing in a field of flowers, shallow depth of field, Canon photography",
    #                   "A cheetah mother nurses her cubs in the tall grass of the Serengeti. The early morning sun beams down through the grass. National Geographic photography by Frans Lanting",
    #                   "A close-up portrait of a bearded viking warrior in a horned helmet. He stares intensely into the distance while holding a battle axe. Dramatic mood lighting, digital oil painting",
    #                   ],
    #         use_container_width=True
    #     )

def main():
    
    st.sidebar.info("**Start here ↓**", icon="👋🏾")

    #two choosen option only
    option_gen = st.sidebar.selectbox(
        "Choose a model", ["Interior", "Exterior"], index=0)
    option_function = st.sidebar.selectbox(
        "Choose a function", ["Generate", "ControlNet", "Inpainting"], index=0)

    #load 2 model types
    if option_gen == "Interior":
        model_path = "../checkpoints/Interior.pt"
    else:
        model_path = "runwayml/stable-diffusion-v1-5"
    
    #take image control
    if option_function == "ControlNet":
        st.write("ControlNet")
        pipe = load_controlnet_model(model_path)
        image_controlnet = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if image_controlnet is not None:
            image_controlnet = Image.open(image_controlnet)
            # image_controlnet.save("controlnet.jpg")
        
    elif option_function == "Inpainting":
        st.write("Inpainting")
        pipe = load_model_inpaint(model_path)
        image_inpainting = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        print(image_inpainting)
        
        if image_inpainting is not None:
            image_inpainting = Image.open(image_inpainting)
            h, w = image_inpainting.size
            
            fill_color = "rgba(255, 255, 255, 0.0)"
            stroke_width = st.number_input("Brush Size",
                                        value=64,
                                        min_value=1,
                                        max_value=100)
            stroke_color = "rgba(255, 255, 255, 1.0)"
            bg_color = "rgba(0, 0, 0, 1.0)"
            drawing_mode = "freedraw"
            
            st.caption(
                "Draw a mask to inpaint, then click the 'Send to Streamlit' button (bottom left, with an arrow on it).")
            canvas_result = st_canvas(
                fill_color=fill_color,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=image_inpainting,
                update_streamlit=False,
                height=h,
                width=w,
                drawing_mode=drawing_mode,
                key="canvas",
            )
            if canvas_result:
                mask = canvas_result.image_data
                mask = mask[:, :, -1] > 0
                if mask.sum() > 0:
                    mask = Image.fromarray(mask)
                    
                      
    else:
        pipe = load_model_base(model_path)
    
    # Form for user input
    with st.sidebar:
        with st.form("generation-form"):
            prompt = st.text_area(
                ":orange[**Enter prompt: ✍🏾**]",
                value="1 cô gái",
                height=150)
            negative_prompt = st.text_area(":orange[**Negative prompt 🙅🏽‍♂️**]",
                                            value="",
                                            help="This is a negative prompt, basically type what you don't want to see in the generated image",
                                            height=100)
            
            st.divider()
            num_outputs = st.slider(
                "Number of images to output", value=1, min_value=1, max_value=4)
            width = st.number_input("Width of output image", value=512)
            height = st.number_input("Height of output image", value=512)

            # The Big Red "Submit" Button!
            submitted = st.form_submit_button(
                "Submit", type="primary", use_container_width=True)
    
    if submitted:
        with st.status('👩🏾‍🍳 Whipping up your words into art...', expanded=True) as status:
            st.write("⚙️ Model initiated")
            st.write("🙆‍♀️ Stand up and strecth in the meantime")
            
        try: 
            if submitted:
                # Calling the replicate API to get the image
                with generated_images_placeholder.container():
                    all_images = []  # List to store all generated images
                    prompt = translate_to_eng(prompt)
                    negative_prompt = translate_to_eng(negative_prompt)
                    seed = random.randint(0, 100000)
                    
                    if option_function == "Generate":
                        output = gen_base(pipe, 
                                        prompt, 
                                        trigger_words="", 
                                        neg=negative_prompt, 
                                        num_images=num_outputs, 
                                        height=height, width=width,
                                        seed = seed)
                    elif option_function == "ControlNet":
                        output = gen_controlnet(pipe, 
                                            prompt, trigger_words="", 
                                            neg=negative_prompt, 
                                            num_images=num_outputs, 
                                            height=height, width=width, 
                                            image=image_controlnet,
                                            seed = seed)
                    else: 
                        output = inpaint_gen(pipe,
                                            image_inpainting,
                                            mask,
                                            prompt,
                                            neg=negative_prompt, 
                                            num_images=num_outputs, 
                                            height=height, width=width,
                                            seed = seed)
                    
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
                    # Save all generated images to session state
                    st.session_state.all_images = all_images
        
        except Exception as e:
            print(e)
            st.error(f'Encountered an error: {e}', icon="🚨")
    

if __name__ == "__main__":
    main()

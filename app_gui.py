import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ExifTags
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import threading
import time

RESIZE_IMAGE = False
PROMPT_FILE = "last_prompt.txt"
START_TIME = time.time()
END_TIME = time.time()

model_name = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16).to("cuda")
processor = AutoProcessor.from_pretrained(
    model_name, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
)


def load_last_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as file:
            return file.read().strip()
    return "Please describe the image in every detail."


def save_last_prompt(prompt):
    with open(PROMPT_FILE, "w", encoding="utf-8") as file:
        file.write(prompt)


LAST_PROMPT = load_last_prompt()
PAD_LEFT = 5
PAD_RIGHT = 45


class ImageDescriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QWEN Image Description")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=0)
        self.root.grid_columnconfigure(3, weight=0)

        self.path_label = tk.Label(root, text="Image Path:")
        self.path_label.grid(
            row=0, column=0, padx=PAD_LEFT, pady=5, sticky="w")

        self.path_entry = tk.Entry(root, width=50)
        self.path_entry.grid(
            row=0, column=1, padx=PAD_LEFT, pady=5, sticky="ew")

        self.browse_button = tk.Button(
            root, text="Browse", command=self.browse_folder)
        self.browse_button.grid(
            row=0, column=2, padx=PAD_LEFT, pady=5, sticky="w")

        self.trigger_label = tk.Label(root, text="Trigger Word:")
        self.trigger_label.grid(
            row=1, column=0, padx=PAD_LEFT, pady=5, sticky="w")

        self.trigger_entry = tk.Entry(root, width=50)
        self.trigger_entry.grid(row=1, column=1, columnspan=2, padx=(
            PAD_LEFT, PAD_RIGHT), pady=5, sticky="ew")

        self.prompt_label = tk.Label(root, text="Prompt:")
        self.prompt_label.grid(
            row=2, column=0, padx=PAD_LEFT, pady=5, sticky="w")

        self.prompt_entry = tk.Text(root, width=50, height=6, wrap="word")
        self.prompt_entry.insert("1.0", LAST_PROMPT)
        self.prompt_entry.grid(row=2, column=1, columnspan=2, padx=(
            PAD_LEFT, PAD_RIGHT), pady=5, sticky="ew")

        self.analyze_button = tk.Button(
            root, text="Analyze Images", command=self.analyze_images)
        self.analyze_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.exit_button = tk.Button(
            root, text="Close", command=self.exit_application)
        self.exit_button.grid(row=4, column=0, columnspan=3, pady=5)

        self.description_text = tk.Text(root, width=70, height=20, wrap="word")
        self.description_text.grid(
            row=5, column=0, columnspan=3, padx=PAD_LEFT, pady=5)

        self.image_label = tk.Label(root)
        self.image_label.grid(row=0, column=3, rowspan=6,
                              padx=PAD_LEFT, pady=5, sticky="nsew")

        self.time_info_label = tk.Label(root, text="", justify="left")
        self.time_info_label.grid(
            row=7, column=0, columnspan=3, pady=5, sticky="w")

    def exit_application(self):
        if hasattr(self, "running_thread") and self.running_thread.is_alive():
            self.running_thread.join(timeout=1)

        self.root.quit()
        self.root.destroy()
        os._exit(0)

    def browse_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select a Folder with Images")
        if folder_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, folder_path)

    def resize_image(self, image_path, max_size=(512, 512)):
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            resized_path = os.path.splitext(image_path)[0] + "_resized.jpg"
            img.save(resized_path)
        return resized_path

    def correct_image_orientation(self, img):
        try:
            exif = img._getexif()
            if exif is not None:
                for orientation_tag, orientation in ExifTags.TAGS.items():
                    if orientation == 'Orientation':
                        orientation_value = exif.get(orientation_tag, 1)
                        if orientation_value == 3:
                            img = img.rotate(180, expand=True)
                        elif orientation_value == 6:
                            img = img.rotate(270, expand=True)
                        elif orientation_value == 8:
                            img = img.rotate(90, expand=True)
        except Exception as e:
            print(f"Error correcting image orientation: {e}")
        return img

    def generate_description(self, image_path):
        global START_TIME

        resized_image_path = self.resize_image(
            image_path) if RESIZE_IMAGE else image_path

        image = Image.open(resized_image_path)

        print(f"Analyzing: {image_path}")
        START_TIME = time.time()

        custom_prompt = self.prompt_entry.get("1.0", "end-1c")
        save_last_prompt(custom_prompt)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": f"{custom_prompt}"},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True)

        inputs = processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        )

        inputs = inputs.to("cuda")

        torch.cuda.empty_cache()

        generated_ids = model.generate(**inputs, max_new_tokens=4096)
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        breakpoint = "\nassistant\n"
        index = output_text[0].index(breakpoint)
        length = len(breakpoint)
        output = output_text[0][index + length:]

        return output

    def analyze_images(self):
        folder_path = self.path_entry.get()
        trigger_word = self.trigger_entry.get()
        if not folder_path:
            messagebox.showerror("Error", "Please enter a folder path.")
            return

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(
            ('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
        if not image_files:
            messagebox.showerror(
                "Error", "No supported images found in the specified folder.")
            return

        self.running_thread = threading.Thread(target=self.process_images, args=(
            folder_path, image_files, trigger_word))
        self.running_thread.start()

    def update_time_info(self, last_duration, average_time, estimated_time_left):
        hours, rem = divmod(estimated_time_left, 3600)
        minutes, seconds = divmod(rem, 60)

        time_info = (
            f"Last Image Duration: {last_duration:.2f} sec\n"
            f"Average Duration: {average_time:.2f} sec/image\n"
            f"Estimated Remaining Time: {int(hours)}h {int(minutes)}m {int(seconds)}s"
        )
        self.root.after(0, self.time_info_label.config, {"text": time_info})

    def process_images(self, folder_path, image_files, trigger_word):
        global START_TIME, END_TIME
        processed_folder = os.path.join(folder_path, "processed")
        os.makedirs(processed_folder, exist_ok=True)

        total_duration = 0
        num_images = len(image_files)

        for i, file_name in enumerate(image_files):
            image_path = os.path.join(folder_path, file_name)
            try:
                START_TIME = time.time()
                description = self.generate_description(image_path)
                END_TIME = time.time()

                if trigger_word:
                    description = f"{trigger_word}, {description}"

                self.show_image(image_path)
                self.update_description(description)

                text_file_path = os.path.splitext(image_path)[0] + ".txt"
                with open(text_file_path, "w", encoding="utf-8") as text_file:
                    text_file.write(description)

                processed_image_path = os.path.join(
                    processed_folder, file_name)
                processed_text_path = os.path.join(
                    processed_folder, os.path.basename(text_file_path))
                shutil.move(image_path, processed_image_path)
                shutil.move(text_file_path, processed_text_path)

                duration = END_TIME - START_TIME
                total_duration += duration
                average_time = total_duration / (i + 1)
                remaining_images = num_images - (i + 1)
                estimated_time_left = remaining_images * average_time

                self.update_time_info(
                    duration, average_time, estimated_time_left)
            except Exception as e:
                print(f"Error with {file_name}: {e}\n")

    def update_description(self, description):
        self.root.after(0, self._clear_and_insert_description, description)

    def _clear_and_insert_description(self, description):
        self.description_text.delete(1.0, tk.END)
        self.description_text.insert(tk.END, description + "\n\n")
        self.description_text.yview(tk.END)

    def show_image(self, image_path):
        img = Image.open(image_path)
        img = self.correct_image_orientation(img)

        size = 720

        if img.width < img.height:
            new_size = (size, int(size * img.height / img.width))
        else:
            new_size = (int(size * img.width / img.height), size)

        img = img.resize(new_size)
        img_tk = ImageTk.PhotoImage(img)

        self.root.after(0, self._update_image_label, img_tk)

    def _update_image_label(self, img_tk):
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk


def main():
    root = tk.Tk()
    app = ImageDescriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

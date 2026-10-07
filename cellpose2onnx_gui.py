import os
import tkinter as tk
from tkinter import filedialog, messagebox
from cellpose2onnx import convert_all_models, convert_to_ONNX


def start_conversion(model_path, output_directory, mean_diameter_str, status_label=None):
    model_path = model_path.strip()
    output_directory = output_directory.strip()
    mean_diameter_str = mean_diameter_str.strip()

    if not output_directory:
        messagebox.showerror("Error", "Output directory must be specified.")
        if status_label:
            status_label.config(text="Error: Output directory missing", fg="red")
        return

    os.makedirs(output_directory, exist_ok=True)

    if model_path:
        if not mean_diameter_str:
            messagebox.showerror(
                "Error",
                "Mean diameter must be provided when converted an individual model."
            )
            if status_label:
                status_label.config(text="Error: Mean diameter missing", fg="red")
            return

        try:
            mean_diameter = float(mean_diameter_str)
        except ValueError:
            messagebox.showerror(
                "Error",
                "Mean diameter must be a valid number (e.g. 17.0 or 30.0)."
            )
            if status_label:
                status_label.config(text="Error: Invalid mean diameter", fg="red")
            return

        if mean_diameter not in [17.0, 30.0]:
            messagebox.showerror(
                "Error",
                "Mean diameter must be either 17.0 (for nuclei-based models) or 30.0 for all other models."
            )
            if status_label:
                status_label.config(text="Error: Invalid mean diameter value", fg="red")
            return

        try:
            if status_label:
                status_label.config(text="Converting individual model...", fg="yellow")
                status_label.update_idletasks()
            convert_to_ONNX(model_path=model_path, output_directory=output_directory, diam_mean=mean_diameter)
            if status_label:
                status_label.config(text="Conversion completed successfully!", fg="green")
            messagebox.showinfo("Info", f"Output model is saved here: {output_directory}\nConversion completed.")
        except Exception as e:
            if status_label:
                status_label.config(text=f"Error during conversion: {str(e)}", fg="red")
            messagebox.showerror("Error", f"Failed to convert model:\n{str(e)}")
    else:
        try:
            if status_label:
                status_label.config(text="Converting all built-in models...", fg="yellow")
                status_label.update_idletasks()
            convert_all_models(output_directory)
            if status_label:
                status_label.config(text="All models converted successfully!", fg="green")
            messagebox.showinfo("Info", f"Output models are saved here: {output_directory}\nConversion completed.")
        except Exception as e:
            if status_label:
                status_label.config(text=f"Error during conversion: {str(e)}", fg="red")
            messagebox.showerror("Error", f"Failed to convert models:\n{str(e)}")


def select_model_path(entry):
    model_path = filedialog.askopenfilename()
    if model_path:
        entry.delete(0, tk.END)
        entry.insert(0, model_path)


def select_output_directory(entry):
    output_directory = filedialog.askdirectory()
    if output_directory:
        entry.delete(0, tk.END)
        entry.insert(0, output_directory)


def create_gui():
    root = tk.Tk()
    root.title("Cellpose to ONNX Converter")
    root.configure(bg='black')

    tk.Label(root, text="Model Path:", fg='white', bg='black').grid(row=0, column=0, padx=10, pady=5, sticky='e')
    model_path_entry = tk.Entry(root, width=50)
    model_path_entry.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=lambda: select_model_path(model_path_entry)).grid(row=0, column=2, padx=10, pady=5)

    tk.Label(root, text="Output Directory:", fg='white', bg='black').grid(row=1, column=0, padx=10, pady=5, sticky='e')
    output_directory_entry = tk.Entry(root, width=50)
    output_directory_entry.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=lambda: select_output_directory(output_directory_entry)).grid(row=1, column=2, padx=10, pady=5)

    tk.Label(root, text="Mean Diameter:", fg='white', bg='black').grid(row=2, column=0, padx=10, pady=5, sticky='e')
    mean_diameter_entry = tk.Entry(root, width=50)
    mean_diameter_entry.grid(row=2, column=1, padx=10, pady=5)
    mean_diameter_entry.insert(0, "30.0")

    status_label = tk.Label(root, text="Ready", fg='white', bg='black')
    status_label.grid(row=3, column=0, columnspan=3, padx=10, pady=5)

    tk.Button(
        root,
        text="Convert",
        command=lambda: start_conversion(
            model_path_entry.get(),
            output_directory_entry.get(),
            mean_diameter_entry.get(),
            status_label
        )
    ).grid(row=4, column=1, padx=10, pady=15)

    root.mainloop()


if __name__ == "__main__":
    create_gui()

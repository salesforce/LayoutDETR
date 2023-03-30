import os
import shutil

def safeMakeDirs(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('Failed to make dirs at {}'.format(dir))


def generate_htmls(generated_image_paths, dummy_html_path, out_dir):
    html_urls = []
    for image_path in generated_image_paths:
        image_basename = os.path.basename(image_path)
        image_name, _ = os.path.splitext(image_basename)
        tgt_html_path = os.path.join(out_dir, f'{image_name}.html')
        shutil.copy(dummy_html_path, tgt_html_path)
        html_urls.append(tgt_html_path)
    return html_urls
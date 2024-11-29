# script_runner.py
import sys
from io import StringIO
from typing import Dict, Optional, Callable
import traceback
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import re
from PySide6.QtWidgets import (QFileDialog, QMessageBox, QWidget, QApplication, 
                              QVBoxLayout, QPushButton)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtCore import QUrl
from tabulate import tabulate
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

class ScriptError:
    def __init__(self, message, traceback=None):
        self.message = message
        self.traceback = traceback
        self.is_script_error = True

class ScriptRunner:
    """Simplified script runner for Python scripts"""
    
    def __init__(self, parent_widget: Optional[QWidget] = None):
        self.logger = logging.getLogger(__name__)
        self.parent_widget = parent_widget  # Store parent widget for dialogs
        # Configure base plotting settings
        self._setup_plotting()
        self._setup_plotly()

    def _setup_plotly(self):
        """Configure Plotly settings"""
        pio.renderers.default = 'browser'

    def _create_plotly_window(self, fig):
        """Create a window to display a Plotly figure"""
        # Create a QWidget container
        container = QWidget()
        container.setWindowTitle("Plotly Plot")
        container.resize(800, 600)
        
        # Create layout
        layout = QVBoxLayout(container)
        
        # Create QWebEngineView for the plot
        web_view = QWebEngineView()
        
        # Convert plotly figure to HTML
        html = fig.to_html(include_plotlyjs='cdn')
        
        # Load the HTML into the web view
        web_view.setHtml(html)
        
        # Add the web view to the layout
        layout.addWidget(web_view)
        
        # Add export button
        export_button = QPushButton("Export Data")
        export_button.clicked.connect(lambda: self._export_plotly_data(fig))
        layout.addWidget(export_button)
        
        container.show()
        return container

    def _export_plotly_data(self, fig):
        """Export Plotly figure data to CSV"""
        try:
            if not fig.data:
                raise ValueError("No data available to export")
            
            # Get data from the figure
            data_dict = {}
            for trace in fig.data:
                if hasattr(trace, 'name') and trace.name:
                    name = trace.name
                else:
                    name = f"Series_{len(data_dict)}"
                    
                data_dict[f"{name}_x"] = trace.x
                data_dict[f"{name}_y"] = trace.y
            
            # Create DataFrame
            df = pd.DataFrame(data_dict)
            
            # Get save path from user
            filepath, _ = QFileDialog.getSaveFileName(
                self.parent_widget,
                "Export Plot Data",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if filepath:
                if not filepath.lower().endswith('.csv'):
                    filepath += '.csv'
                
                df.to_csv(filepath, index=False)
                QMessageBox.information(
                    self.parent_widget,
                    "Success",
                    f"Data exported successfully to:\n{filepath}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self.parent_widget,
                "Error",
                f"Failed to export data: {str(e)}"
            )

    # Keep your existing formatting methods
    def _format_dataframe_output(self, text: str) -> str:
        """Format DataFrame output with proper spacing and borders"""
        try:
            lines = text.strip().split('\n')
            
            if len(lines) > 1 and ('count' in lines[0] or 'mean' in lines[0] or 'std' in lines[0]):
                data = []
                headers = []
                index = []
                
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                        
                    parts = line.split()
                    if not parts:
                        continue
                        
                    if i == 0:  # Header row
                        headers = parts
                    else:  # Data rows
                        index.append(parts[0])
                        row_data = []
                        for val in parts[1:]:  # Skip first column (index)
                            try:
                                row_data.append(float(val))
                            except ValueError:
                                row_data.append(val)
                        data.append(row_data)
                
                if data and headers:
                    df = pd.DataFrame(data, columns=headers[1:], index=index)
                    return tabulate(df, headers='keys', tablefmt='grid', 
                                showindex=True, floatfmt='.6f',
                                numalign='right', stralign='left')
                    
        except Exception as e:
            self.logger.warning(f"Error formatting DataFrame: {str(e)}")
        return text

    def _format_classification_report(self, text: str) -> str:
        """Format classification report with proper alignment"""
        if 'precision    recall  f1-score   support' in text:
            lines = text.split('\n')
            formatted_lines = []
            header_added = False
            
            for line in lines:
                if not line.strip():
                    continue
                    
                if not header_added and 'precision' in line:
                    formatted_lines.append('=' * 80)
                    formatted_lines.append(line)
                    formatted_lines.append('-' * 80)
                    header_added = True
                else:
                    parts = line.split()
                    if len(parts) >= 5:
                        formatted_line = f"{parts[0]:<15} {parts[1]:>10} {parts[2]:>10} {parts[3]:>10} {parts[4]:>10}"
                        formatted_lines.append(formatted_line)
                    else:
                        formatted_lines.append(line)
            
            return '\n'.join(formatted_lines)
        return text

    def _format_output(self, text: str) -> str:
        """Format various types of output text"""
        if not isinstance(text, str):
            return str(text)
        
        if not text.strip():
            return text
            
        if any(marker in text for marker in ['count', 'mean', 'std', 'min', 'max']):
            formatted = self._format_dataframe_output(text)
            if formatted != text:
                return formatted
        
        if 'precision    recall  f1-score   support' in text:
            formatted = self._format_classification_report(text)
            if formatted != text:
                return formatted
        
        if text.strip():
            if '=' * 20 in text:
                return text
            
            if text.endswith('...') or ':' in text:
                return f"\n{'=' * 80}\n{text}"
                
            return text
        
        return text

    def _setup_plotting(self):
        """Configure global plotting settings for compact, readable plots"""
        plt.rcdefaults()
        plt.style.use('default')

        self.save_dpi = 300
        self.save_size = (16, 12)
        
        try:
            from PyQt5.QtWidgets import QApplication
            screen = QApplication.primaryScreen().availableGeometry()
            max_width = min(screen.width() * 0.5, 800)
            max_height = min(screen.height() * 0.5, 600)
        except:
            max_width = 800
            max_height = 600
        
        import matplotlib as mpl
        dpi = mpl.rcParams['figure.dpi']
        width_inches = max_width / dpi
        height_inches = max_height / dpi
        
        sns.set_style("white")
        sns.set_context("paper", rc={"font.size":4, "axes.titlesize":6, "axes.labelsize":4})
        
        plt.rcParams.update({
            'figure.figsize': [width_inches, height_inches],
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'figure.autolayout': True,
            'figure.constrained_layout.use': True,
            'font.size': 4,
            'axes.titlesize': 6,
            'axes.labelsize': 4,
            'xtick.labelsize': 4,
            'ytick.labelsize': 4,
            'legend.fontsize': 4,
            'figure.subplot.left': 0.1,
            'figure.subplot.right': 0.9,
            'figure.subplot.bottom': 0.1,
            'figure.subplot.top': 0.9,
            'figure.subplot.hspace': 0.4,
            'figure.subplot.wspace': 0.3,
        })

    def _clean_script(self, script: str) -> str:
        """Remove markdown code blocks and normalize script"""
        script = re.sub(r'^```python\s*', '', script)
        script = re.sub(r'^```\s*', '', script)
        script = re.sub(r'\s*```$', '', script)
        
        script = script.replace('\r\n', '\n')
        
        lines = script.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():
                cleaned_lines.append(line.rstrip())
        
        return '\n'.join(cleaned_lines)

    def _preprocess_script(self, script: str) -> str:
        """Preprocess script to handle common issues"""
        cleaned_script = self._clean_script(script)
        
        style_commands = {
            "plt.style.use('seaborn')": "sns.set_theme()",
            "plt.style.use('seaborn-v0_8')": "sns.set_theme()",
            "plt.style.use('seaborn-white')": "sns.set_theme(style='white')",
            "plt.style.use('seaborn-dark')": "sns.set_theme(style='dark')",
            "plt.style.use('seaborn-paper')": "sns.set_theme(context='paper')",
            "plt.style.use('seaborn-talk')": "sns.set_theme(context='talk')",
            "plt.style.use('seaborn-poster')": "sns.set_theme(context='poster')",
            "plt.style.use('seaborn-notebook')": "sns.set_theme(context='notebook')",
        }
        
        for old, new in style_commands.items():
            cleaned_script = cleaned_script.replace(old, new)
        
        return cleaned_script

    def export_plot_data(self, fig, ax):
        """Export plot data to CSV"""
        try:
            all_data = {}
            max_length = 0
            
            if hasattr(ax, 'lines_data'):
                for label, data in ax.lines_data.items():
                    max_length = max(max_length, len(data['x']), len(data['y']))
            
            if hasattr(ax, 'scatter_data'):
                for label, data in ax.scatter_data.items():
                    max_length = max(max_length, len(data['x']), len(data['y']))
            
            if hasattr(ax, 'lines_data'):
                for label, data in ax.lines_data.items():
                    x_padded = np.pad(data['x'], (0, max_length - len(data['x'])), 
                                    mode='constant', constant_values=np.nan)
                    y_padded = np.pad(data['y'], (0, max_length - len(data['y'])), 
                                    mode='constant', constant_values=np.nan)
                    all_data[f"{label}_x"] = x_padded
                    all_data[f"{label}_y"] = y_padded
            
            if hasattr(ax, 'scatter_data'):
                for label, data in ax.scatter_data.items():
                    x_padded = np.pad(data['x'], (0, max_length - len(data['x'])), 
                                    mode='constant', constant_values=np.nan)
                    y_padded = np.pad(data['y'], (0, max_length - len(data['y'])), 
                                    mode='constant', constant_values=np.nan)
                    all_data[f"{label}_x"] = x_padded
                    all_data[f"{label}_y"] = y_padded
            
            if not all_data:
                raise ValueError("No data available to export")
            
            df = pd.DataFrame(all_data)
            
            filepath, _ = QFileDialog.getSaveFileName(
                self.parent_widget,
                "Export Plot Data",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if filepath:
                if not filepath.lower().endswith('.csv'):
                    filepath += '.csv'
                
                df.to_csv(filepath, index=False)
                QMessageBox.information(
                    self.parent_widget,
                    "Success",
                    f"Data exported successfully to:\n{filepath}"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self.parent_widget,
                "Error",
                f"Failed to export data: {str(e)}"
            )

    def _adjust_plot(self, fig):
        """Comprehensive plot adjustment including size constraints and text resizing"""
        try:
            for ax in fig.axes:
                lines_data = {}
                for line in ax.get_lines():
                    label = line.get_label() or f"Series_{len(lines_data)}"
                    if label == '_nolegend_':
                        label = f"Series_{len(lines_data)}"
                    x_data = np.array(line.get_xdata())
                    y_data = np.array(line.get_ydata())
                    lines_data[label] = {
                        'x': x_data,
                        'y': y_data
                    }
                
                ax.lines_data = lines_data
                
                scatter_data = {}
                for collection in ax.collections:
                    if hasattr(collection, 'get_offsets'):
                        label = collection.get_label() or f"Scatter_{len(scatter_data)}"
                        if label == '_nolegend_':
                            label = f"Scatter_{len(scatter_data)}"
                        offsets = collection.get_offsets()
                        if len(offsets) > 0:
                            scatter_data[label] = {
                                'x': offsets[:, 0],
                                'y': offsets[:, 1]
                            }
                
                ax.scatter_data = scatter_data
                
                if ax.get_title():
                    ax.set_title(ax.get_title(), size=6, pad=2)
                
                ax.set_xlabel(ax.get_xlabel(), size=4, labelpad=2)
                ax.set_ylabel(ax.get_ylabel(), size=4, labelpad=2)
                
                ax.tick_params(axis='x', labelsize=4, pad=1)
                ax.tick_params(axis='y', labelsize=4, pad=1)
                
                legend = ax.get_legend()
                if legend is not None:
                    plt.setp(legend.get_texts(), fontsize=4)
                    if legend.get_title():
                        plt.setp(legend.get_title(), fontsize=5)
                    legend.set_bbox_to_anchor((1.05, 1))
                    legend.set_frame_on(True)
                
                for text in ax.texts:
                    text.set_fontsize(4)
                
                if hasattr(ax, 'collections'):
                    for collection in ax.collections:
                        if collection.colorbar is not None:
                            collection.colorbar.ax.tick_params(labelsize=4)
                            if collection.colorbar.ax.get_ylabel():
                                collection.colorbar.ax.set_ylabel(
                                    collection.colorbar.ax.get_ylabel(),
                                    size=4
                                )
                
                labels = ax.get_xticklabels()
                if len(labels) > 6:  # If we have many labels
                    for label in labels[::2]:  # Hide every other label
                        label.set_visible(False)
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            fig.tight_layout(pad=0.5)
            
        except Exception as e:
            self.logger.warning(f"Error adjusting plot: {str(e)}")

    def save_plot_high_res(self, fig, filepath=None):
        """Save plot in high resolution"""
        if filepath is None:
            filepath, _ = QFileDialog.getSaveFileName(
                self.parent_widget,
                "Save High Resolution Plot",
                "",
                "PNG Files (*.png);;All Files (*.*)"
            )
            if not filepath:
                return
            if not filepath.lower().endswith('.png'):
                filepath += '.png'

        original_size = fig.get_size_inches()
        original_dpi = fig.get_dpi()
        
        try:
            fig.set_size_inches(self.save_size)
            fig.set_dpi(self.save_dpi)
            
            scale_factor = 2.5
            for ax in fig.axes:
                if ax.get_title():
                    ax.set_title(ax.get_title(), size=6 * scale_factor)
                
                ax.set_xlabel(ax.get_xlabel(), size=4 * scale_factor)
                ax.set_ylabel(ax.get_ylabel(), size=4 * scale_factor)
                
                ax.tick_params(axis='x', labelsize=4 * scale_factor)
                ax.tick_params(axis='y', labelsize=4 * scale_factor)
                
                legend = ax.get_legend()
                if legend is not None:
                    plt.setp(legend.get_texts(), fontsize=4 * scale_factor)
                    if legend.get_title():
                        plt.setp(legend.get_title(), fontsize=5 * scale_factor)
            
            fig.tight_layout()
            
            fig.savefig(filepath, dpi=self.save_dpi, bbox_inches='tight', pad_inches=0.1)
            
            if filepath:
                QMessageBox.information(
                    self.parent_widget,
                    "Success",
                    f"Plot saved successfully to:\n{filepath}"
                )
            
        except Exception as e:
            QMessageBox.critical(
                self.parent_widget,
                "Error",
                f"Failed to save plot: {str(e)}"
            )
            
        finally:
            fig.set_size_inches(original_size)
            fig.set_dpi(original_dpi)
            self._adjust_plot(fig)
            fig.canvas.draw()

    def run_script(self, script: str, local_vars: Dict = None, 
                stdout_callback: Optional[Callable] = None,
                stderr_callback: Optional[Callable] = None) -> Dict:
        """Run script with output capture and formatting"""
        try:
            processed_script = self._preprocess_script(script)
            
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            try:
                class FormattedStdout:
                    def __init__(self, original_stdout, callback, formatter):
                        self.original_stdout = original_stdout
                        self.callback = callback
                        self.formatter = formatter
                        self.buffer = ""

                    def write(self, text):
                        self.buffer += text
                        if '\n' in text:
                            lines = self.buffer.split('\n')
                            self.buffer = lines[-1]
                            complete_lines = lines[:-1]
                            for line in complete_lines:
                                formatted_line = self.formatter(line)
                                if self.callback:
                                    self.callback(formatted_line + '\n')
                                self.original_stdout.write(formatted_line + '\n')

                    def flush(self):
                        if self.buffer:
                            formatted_buffer = self.formatter(self.buffer)
                            if self.callback:
                                self.callback(formatted_buffer)
                            self.original_stdout.write(formatted_buffer)
                            self.buffer = ""
                        self.original_stdout.flush()

                class FormattedStderr:
                    def __init__(self, original_stderr, callback):
                        self.original_stderr = original_stderr
                        self.callback = callback
                        self.buffer = ""

                    def write(self, text):
                        self.buffer += text
                        if '\n' in text:
                            lines = self.buffer.split('\n')
                            self.buffer = lines[-1]
                            complete_lines = lines[:-1]
                            for line in complete_lines:
                                if self.callback:
                                    self.callback(line + '\n')
                                self.original_stderr.write(line + '\n')

                    def flush(self):
                        if self.buffer:
                            if self.callback:
                                self.callback(self.buffer)
                            self.original_stderr.write(self.buffer)
                            self.buffer = ""
                        self.original_stderr.flush()

                formatted_stdout = FormattedStdout(stdout_capture, stdout_callback, self._format_output)
                formatted_stderr = FormattedStderr(stderr_capture, stderr_callback)
                sys.stdout = formatted_stdout
                sys.stderr = formatted_stderr
                
                def _handle_print(*args, **kwargs):
                    for arg in args:
                        if 'plotly.graph_objs' in str(type(arg)):
                            # Handle Plotly figure
                            self._create_plotly_window(arg)
                        else:
                            # Handle regular print
                            message = str(arg)
                            if any(error_term in message.lower() for error_term in ['error', 'exception', 'failed', 'error:']):
                                if stderr_callback:
                                    stderr_callback(message + '\n')
                                formatted_stderr.write(message + '\n')
                            else:
                                formatted_stdout.write(message + '\n')
                
                exec_globals = {
                    '__name__': '__main__',
                    'pd': pd,
                    'np': np,
                    'plt': plt,
                    'sns': sns,
                    'px': px,  # Add plotly express
                    'go': go,  # Add plotly graph objects
                    'print': _handle_print
                }
                if local_vars:
                    exec_globals.update(local_vars)
                
                plt.close('all')
                
                try:
                    exec(processed_script, exec_globals)
                    success = True
                    script_error = None
                except Exception as e:
                    script_error = ScriptError(str(e), traceback.format_exc())
                    if stderr_callback:
                        stderr_callback(str(script_error.message))
                    success = False
                
                if plt.get_fignums():
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        manager = fig.canvas.manager
                        
                        if not hasattr(manager, 'window'):
                            continue
                            
                        manager.window.setGeometry(50, 50, 800, 600)
                        manager.window.setMinimumSize(400, 300)
                        
                        menubar = manager.window.menuBar()
                        menubar.clear()
                        
                        file_menu = menubar.addMenu('File')
                        
                        save_action = file_menu.addAction('Save High Resolution...')
                        save_action.triggered.connect(
                            lambda checked, f=fig: self.save_plot_high_res(f)
                        )
                        
                        for ax in fig.axes:
                            title = ax.get_title() or f"Plot {len(file_menu.actions())}"
                            export_action = file_menu.addAction(f'Export Data ({title})...')
                            export_action.triggered.connect(
                                lambda checked, f=fig, a=ax: self.export_plot_data(f, a)
                            )
                        
                        self._adjust_plot(fig)
                        
                        manager.window.update()
                        
                        fig.show()
                    
                    plt.draw_all()
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                if hasattr(formatted_stdout, 'flush'):
                    formatted_stdout.flush()
                if hasattr(formatted_stderr, 'flush'):
                    formatted_stderr.flush()
            
            return {
                'success': success,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue(),
                'error': script_error if not success else None
            }
            
        except Exception as e:
            app_error = f"Application Error: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(app_error)
            raise RuntimeError(app_error)
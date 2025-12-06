import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Upload, Activity, Zap, Moon, FileText, CheckCircle, AlertCircle, Loader2, Dumbbell, Watch } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface DashboardProps {
    data: any[];
    onUpload: (file: File) => Promise<void>;
}

interface UploadedFile {
    name: string;
    status: 'uploading' | 'processing' | 'success' | 'error';
    type: 'Strong' | 'Whoop' | 'Other';
    date: Date;
}

const Dashboard: React.FC<DashboardProps> = ({ data, onUpload }) => {
    const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            const file = acceptedFiles[0];
            // Simple heuristic to guess type
            let type: 'Strong' | 'Whoop' | 'Other' = 'Other';
            if (file.name.toLowerCase().includes('strong')) type = 'Strong';
            else if (file.name.toLowerCase().includes('whoop')) type = 'Whoop';

            const newFile: UploadedFile = {
                name: file.name,
                status: 'uploading',
                type,
                date: new Date()
            };

            setUploadedFiles(prev => [newFile, ...prev]);
            setIsProcessing(true);

            try {
                await onUpload(file);
                setUploadedFiles(prev => prev.map(f =>
                    f.name === file.name ? { ...f, status: 'success' } : f
                ));
            } catch (error) {
                setUploadedFiles(prev => prev.map(f =>
                    f.name === file.name ? { ...f, status: 'error' } : f
                ));
            } finally {
                setIsProcessing(false);
            }
        }
    }, [onUpload]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] },
        disabled: isProcessing
    });

    // Process data for charts
    const chartData = data.map(d => ({
        date: new Date(d.date).toLocaleDateString(),
        volume: (d.weight_lb || 0) * (d.reps || 0) * (d.sets || 1),
        recovery: d.recovery_score || 0,
        strain: d.day_strain || 0,
        sleep: d.sleep_performance || 0,
    })).reverse();

    return (
        <div className="min-h-screen bg-gray-50/5 text-gray-100 p-8 font-sans">
            <div className="max-w-6xl mx-auto space-y-12">

                {/* Header - Centered & Pop */}
                <header className="text-center space-y-4">
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="inline-flex items-center justify-center p-4 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl shadow-lg mb-2"
                    >
                        <Dumbbell size={40} className="text-white" />
                    </motion.div>
                    <h1 className="text-4xl md:text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400 tracking-tight">
                        Agentic Fitness Bot
                    </h1>
                    <p className="text-lg text-gray-400 max-w-2xl mx-auto">
                        Your AI-powered coach for strength, recovery, and performance analysis.
                    </p>
                </header>

                {/* File Uploads Section */}
                <section className="space-y-6">
                    <div className="flex items-center justify-between">
                        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                            <FileText className="text-blue-500" />
                            File Uploads
                        </h2>
                        <span className="text-sm text-gray-500">Supported: Strong App, Whoop</span>
                    </div>

                    <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden shadow-xl">
                        <div className="grid grid-cols-1 lg:grid-cols-3 divide-y lg:divide-y-0 lg:divide-x divide-gray-800">

                            {/* Dropzone */}
                            <div
                                {...getRootProps()}
                                className={`p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300
                  ${isDragActive ? 'bg-blue-600/10' : 'hover:bg-gray-800/50'}
                  ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
                `}
                            >
                                <input {...getInputProps()} />
                                <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-4 border border-gray-700 shadow-inner">
                                    {isProcessing ? (
                                        <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                                    ) : (
                                        <Upload className="w-8 h-8 text-gray-400" />
                                    )}
                                </div>
                                <h3 className="text-lg font-semibold text-white mb-1">Upload Data</h3>
                                <p className="text-sm text-gray-400">Drag & drop CSV or click to browse</p>
                            </div>

                            {/* File List Table */}
                            <div className="lg:col-span-2 bg-gray-900/50 flex flex-col">
                                <div className="flex-1 overflow-auto min-h-[200px]">
                                    <table className="w-full text-left border-collapse">
                                        <thead>
                                            <tr className="border-b border-gray-800 text-xs uppercase tracking-wider text-gray-500">
                                                <th className="p-4 font-medium">Source</th>
                                                <th className="p-4 font-medium">Filename</th>
                                                <th className="p-4 font-medium">Date</th>
                                                <th className="p-4 font-medium text-right">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody className="text-sm">
                                            {uploadedFiles.length === 0 && (
                                                <tr>
                                                    <td colSpan={4} className="p-8 text-center text-gray-500 italic">
                                                        No files uploaded yet. Start by adding your workout data.
                                                    </td>
                                                </tr>
                                            )}
                                            {uploadedFiles.map((file, idx) => (
                                                <tr key={idx} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                                                    <td className="p-4">
                                                        <div className="flex items-center gap-2">
                                                            {file.type === 'Strong' && <div className="p-1.5 bg-blue-500/20 rounded text-blue-400"><Dumbbell size={14} /></div>}
                                                            {file.type === 'Whoop' && <div className="p-1.5 bg-red-500/20 rounded text-red-400"><Watch size={14} /></div>}
                                                            {file.type === 'Other' && <div className="p-1.5 bg-gray-700/50 rounded text-gray-400"><FileText size={14} /></div>}
                                                            <span className="font-medium text-gray-300">{file.type}</span>
                                                        </div>
                                                    </td>
                                                    <td className="p-4 text-gray-300">{file.name}</td>
                                                    <td className="p-4 text-gray-500">{file.date.toLocaleTimeString()}</td>
                                                    <td className="p-4 text-right">
                                                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border
                              ${file.status === 'success' ? 'bg-green-500/10 text-green-400 border-green-500/20' : ''}
                              ${file.status === 'uploading' ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' : ''}
                              ${file.status === 'error' ? 'bg-red-500/10 text-red-400 border-red-500/20' : ''}
                            `}>
                                                            {file.status === 'success' && <CheckCircle size={12} />}
                                                            {file.status === 'uploading' && <Loader2 size={12} className="animate-spin" />}
                                                            {file.status === 'error' && <AlertCircle size={12} />}
                                                            {file.status.charAt(0).toUpperCase() + file.status.slice(1)}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Stats & Visualization Section */}
                <section className="space-y-6">
                    <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                        <Activity className="text-purple-500" />
                        Performance Insights
                    </h2>

                    {/* KPI Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <StatCard
                            icon={<Dumbbell />}
                            label="Volume Load"
                            value={chartData.length > 0 ? chartData[chartData.length - 1].volume.toLocaleString() : "0"}
                            sub="lbs (last session)"
                            color="blue"
                        />
                        <StatCard
                            icon={<Zap />}
                            label="Recovery"
                            value={chartData.length > 0 ? `${chartData[chartData.length - 1].recovery}%` : "-"}
                            sub="Whoop Score"
                            color="green"
                        />
                        <StatCard
                            icon={<Moon />}
                            label="Sleep Perf"
                            value={chartData.length > 0 ? `${chartData[chartData.length - 1].sleep}%` : "-"}
                            sub="Efficiency"
                            color="purple"
                        />
                    </div>

                    {/* Charts Grid - Horizontal Layout */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <ChartCard title="Volume Progression">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                                    <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickMargin={10} />
                                    <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#fff', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                        itemStyle={{ color: '#E5E7EB' }}
                                    />
                                    <Line type="monotone" dataKey="volume" stroke="#3B82F6" strokeWidth={3} dot={{ r: 4, fill: '#3B82F6' }} activeDot={{ r: 6 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartCard>

                        <ChartCard title="Recovery vs Strain">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                                    <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickMargin={10} />
                                    <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#fff', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                        cursor={{ fill: '#374151', opacity: 0.4 }}
                                    />
                                    <Bar dataKey="recovery" fill="#10B981" radius={[4, 4, 0, 0]} maxBarSize={50} />
                                    <Bar dataKey="strain" fill="#EF4444" radius={[4, 4, 0, 0]} maxBarSize={50} />
                                </BarChart>
                            </ResponsiveContainer>
                        </ChartCard>
                    </div>
                </section>

            </div>
        </div>
    );
};

const StatCard = ({ icon, label, value, sub, color }: any) => {
    const colors: any = {
        blue: "bg-blue-500/10 text-blue-400 border-blue-500/20",
        green: "bg-green-500/10 text-green-400 border-green-500/20",
        purple: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    };

    return (
        <motion.div
            whileHover={{ y: -2 }}
            className={`p-6 rounded-2xl border ${colors[color]} backdrop-blur-sm shadow-lg`}
        >
            <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-white/5">{icon}</div>
                <h3 className="text-sm font-semibold opacity-80 uppercase tracking-wide">{label}</h3>
            </div>
            <div>
                <p className="text-3xl font-bold text-white tracking-tight">{value}</p>
                <span className="text-xs opacity-60 font-medium">{sub}</span>
            </div>
        </motion.div>
    );
};

const ChartCard = ({ title, children }: { title: string, children: React.ReactNode }) => (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-xl h-96 flex flex-col">
        <h3 className="text-lg font-bold text-white mb-6">{title}</h3>
        <div className="flex-1 min-h-0">
            {children}
        </div>
    </div>
);

export default Dashboard;

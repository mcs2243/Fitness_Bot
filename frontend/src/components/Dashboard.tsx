import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import {
    Card,
    CardHeader,
    CardPreview,
    Text,
    Button,
    Title3,
    Body1,
    Caption1,
    Badge,
    TableBody,
    TableCell,
    TableRow,
    Table,
    TableHeader,
    TableHeaderCell,
    Avatar,
    Spinner,
    makeStyles,
    shorthands
} from '@fluentui/react-components';
import {
    ArrowUploadRegular,
    CheckmarkCircleRegular,
    DismissCircleRegular,
    DocumentRegular,
    ActivityRegular,
    WeatherMoonRegular,
    FlashRegular
} from '@fluentui/react-icons';

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

const useStyles = makeStyles({
    container: {
        display: 'flex',
        flexDirection: 'column',
        gap: '24px',
        padding: '32px',
        maxWidth: '1200px',
        margin: '0 auto',
    },
    header: {
        textAlign: 'center',
        marginBottom: '16px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px',
    },
    uploadSection: {
        display: 'grid',
        gridTemplateColumns: '1fr 2fr',
        gap: '24px',
        '@media (max-width: 768px)': {
            gridTemplateColumns: '1fr',
        },
    },
    dropzone: {
        ...shorthands.border('2px', 'dashed', '#424242'),
        ...shorthands.borderRadius('8px'),
        padding: '32px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '16px',
        cursor: 'pointer',
        transition: 'all 0.2s',
        backgroundColor: '#292929',
        ':hover': {
            backgroundColor: '#333333',
            borderColor: '#666666',
        },
    },
    statsGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '16px',
        '@media (max-width: 768px)': {
            gridTemplateColumns: '1fr',
        },
    },
    chartsGrid: {
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '24px',
        '@media (max-width: 768px)': {
            gridTemplateColumns: '1fr',
        },
    },
    chartCard: {
        height: '400px',
        display: 'flex',
        flexDirection: 'column',
    },
    chartContainer: {
        flex: 1,
        minHeight: 0,
        marginTop: '16px',
    },
    brandIcon: {
        width: '32px',
        height: '32px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '4px',
        fontWeight: 'bold',
        fontSize: '10px',
    },
    strongIcon: {
        backgroundColor: '#0078D4',
        color: 'white',
    },
    whoopIcon: {
        backgroundColor: '#D13438',
        color: 'white',
    },
});

const Dashboard: React.FC<DashboardProps> = ({ data, onUpload }) => {
    const styles = useStyles();
    const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            const file = acceptedFiles[0];
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

    const chartData = data.map(d => ({
        date: new Date(d.date).toLocaleDateString(),
        volume: (d.weight_lb || 0) * (d.reps || 0) * (d.sets || 1),
        recovery: d.recovery_score || 0,
        strain: d.day_strain || 0,
        sleep: d.sleep_performance || 0,
    })).reverse();

    return (
        <div className={styles.container}>
            {/* Header */}
            <div className={styles.header}>
                <Title3>Agentic Fitness Bot</Title3>
                <Body1>Your AI-powered coach for strength, recovery, and performance analysis.</Body1>
            </div>

            {/* Data Sources Section */}
            <Card>
                <CardHeader header={<Text weight="semibold">Data Sources</Text>} />
                <div className={styles.uploadSection}>
                    {/* Dropzone */}
                    <div {...getRootProps()} className={styles.dropzone}>
                        <input {...getInputProps()} />
                        {isProcessing ? <Spinner /> : <ArrowUploadRegular fontSize={48} />}
                        <div style={{ textAlign: 'center' }}>
                            <Text weight="semibold">Upload Workout Data</Text>
                            <br />
                            <Caption1>Drag & drop CSV or click to browse</Caption1>
                        </div>
                        <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                            <div className={`${styles.brandIcon} ${styles.strongIcon}`}>Strong</div>
                            <div className={`${styles.brandIcon} ${styles.whoopIcon}`}>Whoop</div>
                        </div>
                    </div>

                    {/* File List */}
                    <div>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHeaderCell>Source</TableHeaderCell>
                                    <TableHeaderCell>Filename</TableHeaderCell>
                                    <TableHeaderCell>Status</TableHeaderCell>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {uploadedFiles.length === 0 && (
                                    <TableRow>
                                        <TableCell colSpan={3} style={{ textAlign: 'center', padding: '24px', color: '#888' }}>
                                            No files uploaded yet.
                                        </TableCell>
                                    </TableRow>
                                )}
                                {uploadedFiles.map((file, idx) => (
                                    <TableRow key={idx}>
                                        <TableCell>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                {file.type === 'Strong' && <div className={`${styles.brandIcon} ${styles.strongIcon}`}>S</div>}
                                                {file.type === 'Whoop' && <div className={`${styles.brandIcon} ${styles.whoopIcon}`}>W</div>}
                                                {file.type === 'Other' && <DocumentRegular />}
                                                <Text>{file.type}</Text>
                                            </div>
                                        </TableCell>
                                        <TableCell><Text>{file.name}</Text></TableCell>
                                        <TableCell>
                                            {file.status === 'success' && <Badge color="success" icon={<CheckmarkCircleRegular />}>Success</Badge>}
                                            {file.status === 'uploading' && <Badge color="brand" icon={<Spinner size="tiny" />}>Uploading</Badge>}
                                            {file.status === 'error' && <Badge color="danger" icon={<DismissCircleRegular />}>Error</Badge>}
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </div>
                </div>
            </Card>

            {/* Stats Grid */}
            <div className={styles.statsGrid}>
                <Card>
                    <CardHeader
                        image={<ActivityRegular fontSize={24} style={{ color: '#0078D4' }} />}
                        header={<Text weight="semibold">Volume Load</Text>}
                        description={<Caption1>Last Session (lbs)</Caption1>}
                    />
                    <Text size={600} weight="bold">
                        {chartData.length > 0 ? chartData[chartData.length - 1].volume.toLocaleString() : "0"}
                    </Text>
                </Card>
                <Card>
                    <CardHeader
                        image={<FlashRegular fontSize={24} style={{ color: '#107C10' }} />}
                        header={<Text weight="semibold">Recovery</Text>}
                        description={<Caption1>Whoop Score</Caption1>}
                    />
                    <Text size={600} weight="bold">
                        {chartData.length > 0 ? `${chartData[chartData.length - 1].recovery}%` : "-"}
                    </Text>
                </Card>
                <Card>
                    <CardHeader
                        image={<WeatherMoonRegular fontSize={24} style={{ color: '#881798' }} />}
                        header={<Text weight="semibold">Sleep Perf</Text>}
                        description={<Caption1>Efficiency</Caption1>}
                    />
                    <Text size={600} weight="bold">
                        {chartData.length > 0 ? `${chartData[chartData.length - 1].sleep}%` : "-"}
                    </Text>
                </Card>
            </div>

            {/* Charts Grid */}
            <div className={styles.chartsGrid}>
                <Card className={styles.chartCard}>
                    <CardHeader header={<Text weight="semibold">Volume Progression</Text>} />
                    <div className={styles.chartContainer}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#424242" vertical={false} />
                                <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickMargin={10} />
                                <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#292929', borderColor: '#424242', color: '#fff' }}
                                    itemStyle={{ color: '#E5E7EB' }}
                                />
                                <Line type="monotone" dataKey="volume" stroke="#0078D4" strokeWidth={3} dot={{ r: 4, fill: '#0078D4' }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                <Card className={styles.chartCard}>
                    <CardHeader header={<Text weight="semibold">Recovery vs Strain</Text>} />
                    <div className={styles.chartContainer}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#424242" vertical={false} />
                                <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickMargin={10} />
                                <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#292929', borderColor: '#424242', color: '#fff' }}
                                    cursor={{ fill: '#424242', opacity: 0.4 }}
                                />
                                <Bar dataKey="recovery" fill="#107C10" radius={[4, 4, 0, 0]} maxBarSize={50} />
                                <Bar dataKey="strain" fill="#D13438" radius={[4, 4, 0, 0]} maxBarSize={50} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </Card>
            </div>
        </div>
    );
};

export default Dashboard;
